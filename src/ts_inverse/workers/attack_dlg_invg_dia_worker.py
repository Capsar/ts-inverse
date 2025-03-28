from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ts_inverse.datahandler import get_mean_std_dataloader, ConcatSliceDataset

from ts_inverse.utils import set_seed, seed_worker
from .attack_worker import AttackWorker, plot_original_and_dummy_data
from .forecasting_worker import evaluate_model


class AttackBaselineWorker(AttackWorker):
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.worker_name = "AttackBaselineWorker"

    def _init_attack_worker_process(self, c, d_c, m_c, fam_c, def_c=None):
        final_model_settings = {**m_c, **fam_c}
        self.train_datasets, self.val_datasets, self.test_datasets = self.get_datasets(**d_c)
        self.g = set_seed(c["seed"])
        train_dataloader = DataLoader(
            ConcatSliceDataset(self.train_datasets),
            batch_size=c["batch_size"],
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.g,
        )

        mean_std_dataloader = DataLoader(
            ConcatSliceDataset(self.train_datasets),
            batch_size=c["batch_size"],
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.g,
        )
        self.inputs_mean, self.inputs_std, self.targets_mean, self.targets_std = get_mean_std_dataloader(
            mean_std_dataloader, c["device"]
        )

        freq_in_day = self.train_datasets[0].freq_in_day
        model_settings = {key: value for key, value in final_model_settings.items() if not key.startswith("_")}
        for key, value in model_settings.items():
            if key.startswith("input_") or key.startswith("output_"):
                model_settings[key] = value * freq_in_day
                final_model_settings[key] = value * freq_in_day

        model = final_model_settings["_model"](**model_settings)  # Create model

        final_model_settings.update(model.extra_info)
        final_config = {**c, **d_c, **final_model_settings}

        print("def_c", def_c)
        if def_c:
            final_config.update({**def_c})

        del final_config["_model"]
        final_config["model"] = model.name
        if "base_num_attack_steps" in final_config:
            final_config["num_attack_steps"] = (
                final_config["base_num_attack_steps"] * final_config["batch_size"] * final_config["_attack_step_multiplier"]
            )
        final_config["train_dataset_size"] = len(train_dataloader.dataset)
        print("Starting attack", final_config["run_number"], "with config:", final_config)
        return model, train_dataloader, final_config

    def worker_process(self, c, d_c, m_c, fam_c, def_c):
        model, train_dataloader, final_config = self._init_attack_worker_process(c, d_c, m_c, fam_c, def_c)
        self.init_logger_object(final_config)
        self.init_attack(model, train_dataloader, final_config)
        self.start_attack(model, config=final_config)
        self._end_logger_object()

    def init_logger_object(self, config):
        tags = [config["attack_method"]]
        project_names = {
            "wandb": "ts-inverse_preparation_baselines",
        }
        self._init_logger_object(project_names, tags, config)

    def init_attack(self, model, tr_dataloader, config):
        self.all_batch_inputs, self.all_batch_targets, self.all_model_state_dicts, self.all_model_gradients, _ = (
            self.train_model_and_record(model, tr_dataloader, config)
        )

        self.all_dummy_inputs, self.all_dummy_targets = self.generate_dummy_data(
            self.all_batch_inputs[0], self.all_batch_targets[0], config
        )

        self.inputs_mean, self.inputs_std = self.inputs_mean[model.features], self.inputs_std[model.features]
        self.targets_mean, self.targets_std = self.targets_mean[0], self.targets_std[0]

    def start_attack(self, model, config):
        for batch_number in range(config["attack_number_of_batches"]):
            model.load_state_dict(self.all_model_state_dicts[batch_number])
            original_dy_dx = self.all_model_gradients[batch_number]
            dummy_inputs = self.all_dummy_inputs[batch_number]
            dummy_targets = self.all_dummy_targets[batch_number]
            batch_inputs = self.all_batch_inputs[batch_number]
            batch_targets = self.all_batch_targets[batch_number]

            self.attack_batch(
                model, config, batch_number, original_dy_dx, dummy_inputs, dummy_targets, batch_inputs, batch_targets
            )

        if config["device"] != "cpu":
            torch.cuda.empty_cache()

        model.eval()

    def attack_batch(self, model, config, batch_number, original_dy_dx, dummy_inputs, dummy_targets, batch_inputs, batch_targets):
        if config["num_attack_steps"] == 0:
            return

        optimization_space = [dummy_inputs, dummy_targets]
        if "TCN" in model.name and config["optimize_dropout"] and config["dropout"] > 0:
            dropout_masks = model.init_dropout_masks(config["device"], config["dropout_mask_init_type"])
            optimization_space += dropout_masks
        dummy_optimizer, dummy_schedular = self.set_attack_optimizer_and_schedular(
            optimization_space, config["dummy_optimizer"], config["learning_rate"], config["lr_decay"], config["num_attack_steps"]
        )

        sample_mapping = np.arange(0, batch_inputs.shape[0])

        plot_original_and_dummy_data(config, sample_mapping, dummy_inputs, dummy_targets, batch_inputs, batch_targets)

        for attack_step in range(0, config["num_attack_steps"] + 1):
            attack_metrics = {
                "step": attack_step,
                "sample_mapping": sample_mapping,
            }

            def closure():
                dummy_optimizer.zero_grad()
                model.zero_grad()  # Should this be done?
                dummy_out = model(dummy_inputs)
                dummy_y = F.mse_loss(dummy_out, dummy_targets)
                dummy_dy_dx = torch.autograd.grad(dummy_y, model.parameters(), create_graph=True)
                if attack_step >= config["num_attack_steps"]:
                    grad_dict, fig = self.plot_gradients(dummy_dy_dx, original_dy_dx, config)
                    attack_metrics.update(grad_dict)
                    self._log_matplotlib_figure(fig, attack_step, log_name="gradients", matplotlib_only=True)

                dy_dx_loss = self.gradient_loss_function(dummy_dy_dx, original_dy_dx, config["gradient_loss"])
                if "total_variation_alpha_inputs" in config and config["total_variation_alpha_inputs"] > 0:
                    dy_dx_loss += config["total_variation_alpha_inputs"] * total_variation_time_series(dummy_inputs)
                if "total_variation_beta_targets" in config and config["total_variation_beta_targets"] > 0:
                    dy_dx_loss += config["total_variation_beta_targets"] * total_variation_time_series(
                        dummy_targets.unsqueeze(-1)
                    )

                if "TCN" in model.name and config["optimize_dropout"] and config["dropout"] > 0:
                    if config["dropout_probability_regularizer"] > 0:
                        for dropout_layer in model.get_dropout_layers():
                            dy_dx_loss += (
                                config["dropout_probability_regularizer"]
                                * ((1 - dropout_layer.do_mask.mean()) - dropout_layer.p).abs()
                            )

                dy_dx_loss.backward()

                if config["use_grad_signs"]:
                    dummy_inputs.grad.sign_()

                return dy_dx_loss

            dy_dx_loss = dummy_optimizer.step(closure)

            self.after_effect(config, model, dummy_inputs, dummy_targets, attack_step)

            self.schedular_step(config["lr_decay"], dummy_schedular, attack_metrics, dy_dx_loss)

            # Should calcualte evalaution metrics and log them
            if attack_step % (config["num_attack_steps"] // min(config["num_attack_steps"], 500)) == 0:
                attack_metrics["grad_diff_loss_mse"] = dy_dx_loss.item()
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
                )
        print("finished attack")

    def schedular_step(self, config_lr_decay, dummy_schedular, attack_metrics, dy_dx_loss):
        if dummy_schedular is not None:
            if "on_plateau" in config_lr_decay:
                dummy_schedular.step(dy_dx_loss)
            else:
                dummy_schedular.step()
            for i, lr in enumerate(dummy_schedular.get_last_lr()):
                attack_metrics[f"learning_rates/lr_{i}"] = lr


def total_variation_time_series(x):
    diffs = (x[:, 1:, :] - x[:, :-1, :]).abs().mean()
    return diffs


# def total_variation(x):
#     """Anisotropic TV. from https://github.com/JonasGeiping/invertinggradients/blob/master/inversefed/metrics.py"""
#     dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
#     dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
#     return dx + dy
