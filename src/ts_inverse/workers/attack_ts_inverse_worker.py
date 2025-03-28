from matplotlib import pyplot as plt
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F

from ts_inverse.attack_time_series_utils import (
    interpolate,
    periodicity_regularization,
    pinball_loss,
    temporal_resolution_warping,
    trend_consistency_regularization,
)
from ts_inverse.models.grad_to_input import ImprovedGradToInputNN_Quantile
from .attack_dlg_invg_dia_worker import total_variation_time_series
from .attack_learning_to_invert_worker import AttackLearningToInvertWorker
from ts_inverse.models import ImprovedGradToInputNN_Probabilistic

from scipy.optimize import linear_sum_assignment

from .attack_worker import get_batch_sample_mapping, plot_original_and_dummy_data


class AttackTSInverseWorker(AttackLearningToInvertWorker):
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.worker_name = "AttackTSInverseWorker"

    def worker_process(self, c, d_c, m_c, fam_c, def_c):
        model, train_dataloader, final_config = self._init_attack_worker_process(c, d_c, m_c, fam_c, def_c)
        self.init_logger_object(final_config)
        self.init_attack(model, train_dataloader, final_config)
        self.start_attack(model, config=final_config)
        self._end_logger_object()

    def init_logger_object(self, config):
        tags = [config["attack_method"]]
        project_names = {
            "wandb": "ts-inverse_preparation",
        }
        return self._init_logger_object(project_names, tags, config)

    def attack_batch(self, model, config, batch_number, original_dy_dx, dummy_inputs, dummy_targets, batch_inputs, batch_targets):
        # Get the inversion model from super class.
        inversion_model = super().attack_batch(
            model, config, batch_number, original_dy_dx, dummy_inputs, dummy_targets, batch_inputs, batch_targets
        )

        if config["num_attack_steps"] == 0:
            return

        if config["batch_size"] == 1 and config["one_shot_targets"]:
            with torch.no_grad():
                grad_w = original_dy_dx[-2]
                grad_b = original_dy_dx[-1].unsqueeze(-1)
                reconstructed_head_inputs = torch.mm(torch.pinverse(grad_b), grad_w)
                dummy_targets = model.fc(reconstructed_head_inputs) - (grad_b.T / (2 / batch_targets.shape[1]))
                dummy_targets.detach().clone()
                print(dummy_inputs.shape, dummy_targets.shape)

        if inversion_model is not None:
            inversion_model.eval()  # To be used for inference here
            flattened_original_dy_dx = torch.cat([g.view(-1) for g in original_dy_dx]).to(config["device"])
            if config["attack_loss"] == "quantile":
                dummy_quantile_inputs, predicted_dummy_quantile_targets = inversion_model.inference(
                    flattened_original_dy_dx.unsqueeze(0)
                )
                regularization_inputs = dummy_quantile_inputs.squeeze(0)  # remove batch dimension
                regularization_targets = predicted_dummy_quantile_targets.squeeze(0)  # remove batch dimension
                # Repeat regularization inputs and targets to match the batch size
                regularization_inputs = regularization_inputs.repeat(
                    batch_inputs.shape[0] // config["inversion_batch_size"], 1, 1, 1
                )
                regularization_targets = regularization_targets.repeat(
                    batch_targets.shape[0] // config["inversion_batch_size"], 1, 1
                )

                # The regularized inputs, targets last dimension are the quantiles and these should be swaped with the batch dimension
                # dummy_inputs = regularization_inputs.permute(3, 1, 2, 0).squeeze(-1).detach().requires_grad_(True)
                # dummy_targets = regularization_targets.permute(2, 1, 0).squeeze(-1).detach().requires_grad_(True)

                # dummy_inputs = regularization_inputs.clone().reshape(batch_inputs.size()).detach().requires_grad_(True)
                # dummy_targets = regularization_targets.clone().reshape(batch_targets.size()).detach().requires_grad_(True)
                # print(regularization_inputs.shape, regularization_targets.shape, batch_inputs.shape, batch_targets.shape)
            else:
                regularization_inputs, regularization_targets = inversion_model(flattened_original_dy_dx.unsqueeze(0))
                regularization_inputs = regularization_inputs.view(batch_inputs.size()).detach()
                regularization_targets = regularization_targets.view(batch_targets.size()).detach()
                dummy_inputs = regularization_inputs.detach().clone().requires_grad_(True)
                dummy_targets = regularization_targets.detach().clone().requires_grad_(True)

        optimization_space = [dummy_inputs]
        if config["batch_size"] > 1 or not config["one_shot_targets"]:
            optimization_space += [dummy_targets]
        if "TCN" in model.name and config["optimize_dropout"] and config["dropout"] > 0:
            dropout_masks = model.init_dropout_masks(config["device"], config["dropout_mask_init_type"])
            optimization_space += dropout_masks

        dummy_optimizer, dummy_schedular = self.set_attack_optimizer_and_schedular(
            optimization_space,
            config["attack_opti_optimizer"],
            config["optimization_learning_rate"],
            config["attack_opti_lr_decay"],
            config["num_attack_steps"],
        )

        sample_mapping = np.arange(0, batch_inputs.shape[0])
        plot_original_and_dummy_data(config, sample_mapping, dummy_inputs, dummy_targets, batch_inputs, batch_targets)

        for attack_step in range(0, config["num_attack_steps"] + 1):
            attack_metrics = {
                "step": attack_step + config["num_learn_epochs"],
                "sample_mapping": sample_mapping,
            }

            def closure():
                dummy_optimizer.zero_grad()
                model.zero_grad()
                dy_dx_loss = torch.zeros(1, device=dummy_inputs.device)

                dummy_out = model(dummy_inputs)
                dummy_y = F.mse_loss(dummy_out, dummy_targets)
                dummy_dy_dx = torch.autograd.grad(dummy_y, model.parameters(), create_graph=True)

                if attack_step >= config["num_attack_steps"]:
                    grad_dict, fig = self.plot_gradients(dummy_dy_dx, original_dy_dx, config)
                    attack_metrics.update(grad_dict)
                    self._log_matplotlib_figure(
                        fig, attack_step + config["num_learn_epochs"], log_name="gradients", matplotlib_only=True
                    )

                dy_dx_loss += self.gradient_loss_function(dummy_dy_dx, original_dy_dx, config["gradient_loss"])

                if (
                    "TCN" in model.name
                    and config["optimize_dropout"]
                    and config["dropout"] > 0
                    and config["dropout_probability_regularizer"] > 0
                ):
                    for dropout_layer in model.get_dropout_layers():
                        dy_dx_loss += (
                            config["dropout_probability_regularizer"]
                            * ((1 - dropout_layer.do_mask.mean()) - dropout_layer.p).abs()
                        )

                if inversion_model is not None and (
                    config["inversion_regularization_term_inputs"] > 0 or config["inversion_regularization_term_targets"] > 0
                ):
                    dy_dx_loss += self.learned_prior_regularization(
                        dummy_inputs, dummy_targets, regularization_inputs, regularization_targets, config
                    )

                if "total_variation_alpha_inputs" in config and config["total_variation_alpha_inputs"] > 0:
                    dy_dx_loss += config["total_variation_alpha_inputs"] * total_variation_time_series(dummy_inputs)
                if "total_variation_beta_targets" in config and config["total_variation_beta_targets"] > 0:
                    dy_dx_loss += config["total_variation_beta_targets"] * total_variation_time_series(
                        dummy_targets.unsqueeze(-1)
                    )

                combined_dummy_data_first_feature = torch.cat([dummy_inputs[:, :, 0], dummy_targets[:, :]], dim=1)  # Same series
                if "lower_res_term_inputs" in config and config["lower_res_term_inputs"] > 0:
                    with torch.no_grad():
                        warped_inputs = temporal_resolution_warping(dummy_inputs, 2)
                        filtered_inputs = interpolate(warped_inputs, dummy_inputs.shape[1])
                    dy_dx_loss += config["lower_res_term_inputs"] * F.l1_loss(filtered_inputs, dummy_inputs)

                if "lower_res_term_targets" in config and config["lower_res_term_targets"] > 0:
                    with torch.no_grad():
                        warped_targets = temporal_resolution_warping(dummy_targets.unsqueeze(-1), 2)
                        filtered_targets = interpolate(warped_targets, dummy_targets.shape[1]).squeeze(-1)
                    dy_dx_loss += config["lower_res_term_targets"] * F.l1_loss(filtered_targets, dummy_targets)

                if "trend_term" in config and config["trend_term"] > 0:
                    lr_term = (
                        dummy_schedular.get_last_lr()[0] / config["optimization_learning_rate"]
                        if attack_step > 0 and config["trend_reduce_lr"]
                        else 1
                    )
                    dy_dx_loss += (
                        lr_term
                        * config["trend_term"]
                        * trend_consistency_regularization(combined_dummy_data_first_feature, config["trend_loss"])
                    )

                if "periodicity_term" in config and config["periodicity_term"] > 0:
                    lr_term = (
                        dummy_schedular.get_last_lr()[0] / config["optimization_learning_rate"]
                        if attack_step > 0 and config["trend_reduce_lr"]
                        else 1
                    )
                    dy_dx_loss += (
                        lr_term
                        * config["periodicity_term"]
                        * periodicity_regularization(
                            combined_dummy_data_first_feature, period=dummy_targets.shape[1], loss=config["periodicity_loss"]
                        )
                    )

                dy_dx_loss.backward()

                if config["grad_signs_for_inputs"]:
                    dummy_inputs.grad.sign_()
                if config["grad_signs_for_targets"]:
                    dummy_targets.grad.sign_()
                if (
                    "TCN" in model.name
                    and config["optimize_dropout"]
                    and config["dropout"] > 0
                    and config["grad_signs_for_dropouts"]
                ):
                    for dropout_layer in model.get_dropout_layers():
                        dropout_layer.do_mask.grad.sign_()

                return dy_dx_loss

            dy_dx_loss = dummy_optimizer.step(closure)

            self.after_effect(config, model, dummy_inputs, dummy_targets, attack_step)

            self.schedular_step(config["attack_opti_lr_decay"], dummy_schedular, attack_metrics, dy_dx_loss)

            # Should calcualte evalaution metrics and log them
            if attack_step % (config["num_attack_steps"] // min(config["num_attack_steps"], 200)) == 0:
                attack_metrics["grad_diff_loss_mse"] = dy_dx_loss
                self.evaluate_and_log_reconstruction(
                    config,
                    batch_inputs,
                    batch_targets,
                    dummy_inputs,
                    dummy_targets,
                    batch_number,
                    attack_step,
                    config["num_attack_steps"],
                    attack_metrics,
                    attack_step_offset=config["num_learn_epochs"],
                )

    def learned_prior_regularization(self, dummy_inputs, dummy_targets, regularization_inputs, regularization_targets, config):
        learned_prior_regularization = torch.zeros(1, device=dummy_inputs.device)
        if config["attack_loss"] == "quantile":
            if config["inversion_regularization_loss"] == "quantile":
                # Make sure the regularization_inputs, which contains the quantile predictions, has the same batch_size as the dummy_inputs
                if config["inversion_regularization_term_inputs"] > 0:
                    reshaped_reg_inputs = regularization_inputs.view(config["batch_size"], -1, len(config["quantiles"]))
                    learned_prior_regularization += config["inversion_regularization_term_inputs"] * pinball_loss(
                        reshaped_reg_inputs, dummy_inputs.view(config["batch_size"], -1), config["quantiles"]
                    )
                if config["inversion_regularization_term_targets"] > 0:
                    # behind quantiles
                    reshaped_reg_targets = regularization_targets.unsqueeze(-2).view(
                        config["batch_size"], -1, len(config["quantiles"])
                    )
                    learned_prior_regularization += config["inversion_regularization_term_targets"] * pinball_loss(
                        reshaped_reg_targets, dummy_targets.view(config["batch_size"], -1), config["quantiles"]
                    )
            if config["inversion_regularization_loss"] == "quantile_bounds":

                def out_of_bound_loss(sequence, sequence_quantiles):
                    bound_loss = torch.zeros(1, device=sequence.device)
                    for tau_q in range(sequence_quantiles.shape[-1] // 2):
                        quantile_upper_bound = sequence_quantiles[..., tau_q].reshape(sequence.shape)
                        quantile_lower_bound = sequence_quantiles[..., -tau_q - 1].reshape(sequence.shape)
                        bound_loss += F.relu(sequence - quantile_upper_bound).mean()
                        bound_loss += F.relu(quantile_lower_bound - sequence).mean()
                    return bound_loss / 2

                if config["inversion_regularization_term_inputs"] > 0:
                    learned_prior_regularization += config["inversion_regularization_term_inputs"] * out_of_bound_loss(
                        dummy_inputs, regularization_inputs
                    )
                if config["inversion_regularization_term_targets"] > 0:
                    learned_prior_regularization += config["inversion_regularization_term_targets"] * out_of_bound_loss(
                        dummy_targets, regularization_targets
                    )

        elif config["inversion_regularization_loss"] == "l1":
            learned_prior_regularization += config["inversion_regularization_term_inputs"] * F.l1_loss(
                dummy_inputs, regularization_inputs
            )
            learned_prior_regularization += config["inversion_regularization_term_targets"] * F.l1_loss(
                dummy_targets, regularization_targets
            )
        else:
            raise NotImplementedError(f"Inversion regularization loss not found: {config["inversion_regularization_loss"]}")
        return learned_prior_regularization

    def initialize_inversion_model(self, config, batch_inputs, batch_targets):
        inversion_model, input_shape, target_shape = super().initialize_inversion_model(config, batch_inputs, batch_targets)
        if inversion_model is not None:
            return inversion_model, input_shape, target_shape
        elif config["inversion_model"] == "ImprovedGradToInputNN_Probabilistic":
            inversion_model = ImprovedGradToInputNN_Probabilistic(
                config["attack_hidden_size"], config["model_size"], input_shape, target_shape, distribution=config["attack_loss"]
            ).to(config["device"])
        elif config["inversion_model"] == "ImprovedGradToInputNN_Quantile":
            inversion_model = ImprovedGradToInputNN_Quantile(
                config["attack_hidden_size"], config["model_size"], input_shape, target_shape, quantiles=config["quantiles"]
            ).to(config["device"])
        return inversion_model, input_shape, target_shape

    def calculate_inversion_model_loss(
        self, inversion_model, config, attack_batch_size, predicted_inputs, predicted_targets, aux_inputs, aux_targets
    ):
        loss = super().calculate_inversion_model_loss(
            inversion_model, config, attack_batch_size, predicted_inputs, predicted_targets, aux_inputs, aux_targets
        )
        if loss != 0:
            return loss

        if config["attack_loss"] == "quantile":
            assert predicted_inputs is not None, "Predicted inputs should not be None for quantile loss"

            predicted_inputs = predicted_inputs.repeat(1, config["batch_size"] // config["inversion_batch_size"], 1, 1, 1)
            predicted_targets = predicted_targets.unsqueeze(-2)  # behind quantiles
            predicted_targets = predicted_targets.repeat(1, config["batch_size"] // config["inversion_batch_size"], 1, 1, 1)
            aux_targets = aux_targets.unsqueeze(-1)  # last dimension

            # Calculate pairwise pinball losses between combined vectors
            # Dimensions of Predicted data = (attack_batch_size, config['batch_size'], sequence_length, n_features, n_quantiles)
            # Dimensions of Auxiliary data = (attack_batch_size, config['batch_size'], sequence_length, n_features)
            if config["inversion_batch_size"] != config["batch_size"] or (
                config["inversion_batch_size"] == config["batch_size"] and config["batch_size"] == 1
            ):
                predicted_inputs = predicted_inputs.view(attack_batch_size * config["batch_size"], -1, len(config["quantiles"]))
                aux_inputs = aux_inputs.view(attack_batch_size * config["batch_size"], -1)
                predicted_targets = predicted_targets.view(attack_batch_size * config["batch_size"], -1, len(config["quantiles"]))
                aux_targets = aux_targets.view(attack_batch_size * config["batch_size"], -1)

                combined_predicteds = torch.cat([predicted_inputs, predicted_targets], dim=1)
                combined_auxes = torch.cat([aux_inputs, aux_targets], dim=1)

                return pinball_loss(combined_predicteds, combined_auxes, config["quantiles"])
            else:
                # This would generate 4 individual quantile predictions.
                batch_wise_combined_loss = torch.empty(
                    (attack_batch_size, config["batch_size"], config["batch_size"]), device=config["device"]
                )
                for i in range(attack_batch_size):
                    for j in range(config["batch_size"]):
                        for k in range(config["batch_size"]):
                            inputs_loss = pinball_loss(predicted_inputs[i, j], aux_inputs[i, k], config["quantiles"])
                            targets_loss = pinball_loss(predicted_targets[i, j], aux_targets[i, k], config["quantiles"])
                            batch_wise_combined_loss[i, j, k] = (inputs_loss + targets_loss) / 2
        elif config["attack_loss"] == "normal" or config["attack_loss"] == "cauchy" or config["attack_loss"] == "beta":
            batch_wise_combined_loss = inversion_model.calculate_batchwise_prob_loss(
                predicted_inputs, predicted_targets, aux_inputs, aux_targets, config
            )
        else:
            raise ValueError(f"Invalid attack loss function: {config['attack_loss']}")

        # Solve the optimal assignment problem for the combined loss matrix
        for combined_loss_matrix in batch_wise_combined_loss:
            row_ind, col_ind = linear_sum_assignment(combined_loss_matrix.detach().cpu().numpy())
            loss += combined_loss_matrix[row_ind, col_ind].mean()

        # Normalize the loss by the attack batch size
        loss /= attack_batch_size

        return loss

    def evaluate_dummy_prediction(
        self,
        config,
        batch_number,
        original_dy_dx,
        batch_inputs,
        batch_targets,
        dummy_inputs,
        dummy_targets,
        inversion_model,
        epoch,
        attack_metrics,
    ):
        if config["attack_loss"] == "quantile":
            inversion_model.eval()
            with torch.no_grad():
                flattened_original_dy_dx = torch.cat([g.view(-1) for g in original_dy_dx]).to(config["device"])
                dummy_quantile_inputs, predicted_dummy_quantile_targets = inversion_model.inference(
                    flattened_original_dy_dx.unsqueeze(0)
                )
                dummy_quantile_inputs = dummy_quantile_inputs.squeeze(0).repeat(
                    batch_inputs.shape[0] // config["inversion_batch_size"], 1, 1, 1
                )
                dummy_inputs = dummy_quantile_inputs

                if predicted_dummy_quantile_targets is not None:
                    dummy_quantile_targets = predicted_dummy_quantile_targets.squeeze(0).repeat(
                        batch_targets.shape[0] // config["inversion_batch_size"], 1, 1
                    )
                    dummy_targets = dummy_quantile_targets

                def pinball_loss_simple(batch_inputs, dummy_inputs):
                    return pinball_loss(dummy_inputs, batch_inputs, config["quantiles"])

                standard_mapping = np.arange(0, batch_inputs.shape[0])
                input_sample_mapping = get_batch_sample_mapping(batch_inputs, dummy_inputs, function=pinball_loss_simple)
                target_sample_mapping = get_batch_sample_mapping(batch_targets, dummy_targets, function=pinball_loss_simple)

                sample_mapping = np.arange(0, batch_inputs.shape[0])
                if not (standard_mapping == input_sample_mapping).all():
                    sample_mapping = input_sample_mapping
                if (standard_mapping == input_sample_mapping).all() and not (standard_mapping == target_sample_mapping).all():
                    sample_mapping = target_sample_mapping

                mean_evaluation = {
                    "inputs/pinball/mean": pinball_loss_simple(batch_inputs, dummy_inputs[sample_mapping]).item(),
                    "targets/pinball/mean": pinball_loss_simple(batch_targets, dummy_targets[sample_mapping]).item(),
                }
                attack_metrics.update(mean_evaluation)

                # individual batch sample metrics
                for i, j in enumerate(sample_mapping):
                    individual_evaluation = {
                        f"inputs/pinball/{i}": pinball_loss_simple(batch_inputs[i], dummy_inputs[j]).item(),
                        f"targets/pinball/{i}": pinball_loss_simple(batch_targets[i], dummy_targets[j]).item(),
                    }
                    attack_metrics.update(individual_evaluation)

                if epoch % (config["num_learn_epochs"] // 10) == 0:
                    quantile_df, fig = plot_quantile_dummy_data(
                        config, sample_mapping, dummy_inputs, dummy_targets, batch_inputs, batch_targets
                    )

                    self._log_dataframe(quantile_df, step=epoch, log_name=f"_quantile_batch_{batch_number}")
                    self._log_matplotlib_figure(fig, step=epoch, log_name=f"_quantile_batch_{batch_number}", matplotlib_only=True)
                self._log_metrics(attack_metrics, step=epoch)
        else:
            super().evaluate_dummy_prediction(
                config,
                batch_number,
                original_dy_dx,
                batch_inputs,
                batch_targets,
                dummy_inputs,
                dummy_targets,
                inversion_model,
                epoch,
                attack_metrics,
            )


def plot_quantile_dummy_data(config, sample_mapping, dummy_quantile_inputs, dummy_quantile_targets, batch_inputs, batch_targets):
    batch_size = batch_inputs.shape[0]
    num_quantiles = dummy_quantile_inputs.shape[3]  # Number of quantiles is in the last dimension for dummy data
    original_x_axis = np.arange(0, batch_inputs.shape[1] + batch_targets.shape[1])
    fig, axes = plt.subplots(nrows=batch_size, figsize=(20, 3.5 * batch_size), sharex=True)
    axes = np.array(axes).reshape(-1)  # Ensure axes is always iterable

    for i, j in enumerate(sample_mapping):
        ax = axes[i]
        # Plotting for batch inputs (no quantiles)
        batch_input_values = batch_inputs[i, :].detach().cpu().numpy()  # Squeeze out the singleton dimension for plotting
        ax.plot(original_x_axis[: batch_inputs.shape[1]], batch_input_values, label=f"Batch Input ({i})")

        # Plotting quantiles for dummy inputs
        for q in range(num_quantiles):
            dummy_input_quantiles = dummy_quantile_inputs[j, :, 0, q].detach().cpu().numpy()
            ax.plot(original_x_axis[: batch_inputs.shape[1]], dummy_input_quantiles, label=f"Dummy Input Quantile {q} ({j})")

        # Plotting for batch targets (assuming no quantiles)
        batch_target_values = batch_targets[i, :].detach().cpu().numpy()
        ax.plot(original_x_axis[batch_inputs.shape[1] :], batch_target_values, label=f"Batch Target ({i})")

        # Plotting quantiles for dummy targets
        for q in range(num_quantiles):
            dummy_target_quantiles = dummy_quantile_targets[j, :, q].detach().cpu().numpy()
            ax.plot(original_x_axis[batch_inputs.shape[1] :], dummy_target_quantiles, label=f"Dummy Target Quantile {q} ({j})")

        ax.set_xlabel("Time" if i == batch_size - 1 else "")
        ax.set_ylabel("Output")
        ax.set_title(
            f'Dataset {config["dataset"]}: Comparison of Inputs & Targets between Dataset & Recovery - Batch Sample {i} & Dummy Sample {j}'
        )
        ax.legend()

    # Create weights and biases plot & dataframe
    fill_nan = np.empty(batch_targets.shape[1])
    fill_nan.fill(np.nan)
    if len(batch_inputs.shape) == 3:
        batch_inputs_dict = {
            f"batch_inputs_{i}_{f}": np.concatenate((batch_inputs[i, :, f].detach().cpu().numpy(), fill_nan))
            for i in range(batch_inputs.shape[0])
            for f in range(batch_inputs.shape[2])
        }
        dummy_quantile_inputs_dict = {
            f"dummy_quantile_inputs_{i}_{f}_{q}": np.concatenate(
                (dummy_quantile_inputs[i, :, f, q].detach().cpu().numpy(), fill_nan)
            )
            for i in range(dummy_quantile_inputs.shape[0])
            for f in range(dummy_quantile_inputs.shape[2])
            for q in range(dummy_quantile_inputs.shape[3])
        }
    elif len(batch_inputs.shape) == 2:
        batch_inputs_dict = {
            f"batch_inputs_{i}": np.concatenate((batch_inputs[i, :].detach().cpu().numpy(), fill_nan))
            for i in range(batch_inputs.shape[0])
        }
        dummy_quantile_inputs_dict = {
            f"dummy_quantile_inputs_{i}_{q}": np.concatenate((dummy_quantile_inputs[i, :, q].detach().cpu().numpy(), fill_nan))
            for i in range(dummy_quantile_inputs.shape[0])
            for q in range(dummy_quantile_inputs.shape[2])
        }

    fill_nan = np.empty(batch_inputs.shape[1])
    fill_nan.fill(np.nan)
    batch_targets_dict = {
        f"batch_targets_{i}": np.concatenate((fill_nan, batch_targets[i, :].detach().cpu().numpy()))
        for i in range(batch_targets.shape[0])
    }
    dummy_quantile_targets_dict = {
        f"dummy_quantile_targets_{i}_{q}": np.concatenate((fill_nan, dummy_quantile_targets[i, :, q].detach().cpu().numpy()))
        for i in range(dummy_quantile_targets.shape[0])
        for q in range(dummy_quantile_targets.shape[2])
    }
    input_data = {**batch_inputs_dict, **dummy_quantile_inputs_dict, **batch_targets_dict, **dummy_quantile_targets_dict}
    df = pd.DataFrame(input_data)

    if config["verbose"]:
        plt.show()
    else:
        plt.close(fig)

    return df, fig
