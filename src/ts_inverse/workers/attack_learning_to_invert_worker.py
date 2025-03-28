import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

from ts_inverse.attack_time_series_utils import interpolate
from ts_inverse.models.grad_to_input import ImprovedGradToInputNN, ImprovedGradToInputNN_2
from ts_inverse.utils import seed_worker
from ts_inverse.models import GradToInputNN

from scipy.optimize import linear_sum_assignment

from .attack_dlg_invg_dia_worker import AttackBaselineWorker
from .attack_worker import apply_sign_transformation, apply_pruning, add_gaussian_noise


class AttackLearningToInvertWorker(AttackBaselineWorker):
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.worker_name = "AttackLearningToInvertWorker"

    def worker_process(self, c, d_c, m_c, fam_c, def_c):
        model, train_dataloader, final_config = self._init_attack_worker_process(c, d_c, m_c, fam_c, def_c)
        self.init_logger_object(final_config)
        self.init_attack(model, train_dataloader, final_config)
        self.start_attack(model, config=final_config)
        self._end_logger_object()

    def init_attack(self, model, tr_dataloader, config):
        super().init_attack(model, tr_dataloader, config)
        config["model_size"] = sum(p.numel() for p in model.parameters())

        if "aux_dataset" in config and config["aux_dataset"] is not None:
            aux_dataset_config = config["aux_dataset"]
            aux_trainset_path = f"../data/_aux_datasets/train_{aux_dataset_config['dataset']}_{len(aux_dataset_config['columns'])}_{aux_dataset_config['train_stride']}_{aux_dataset_config['observation_days']}_{aux_dataset_config['future_days']}_{aux_dataset_config['normalize']}.pt"
            aux_valset_path = f"../data/_aux_datasets/val_{aux_dataset_config['dataset']}_{len(aux_dataset_config['columns'])}_{aux_dataset_config['train_stride']}_{aux_dataset_config['observation_days']}_{aux_dataset_config['future_days']}_{aux_dataset_config['normalize']}.pt"
            if os.path.exists(aux_trainset_path) and os.path.exists(aux_valset_path):
                aux_trainset = torch.load(aux_trainset_path)
                aux_valset = torch.load(aux_valset_path)
            else:
                train_sets, val_sets, test_sets = self.get_datasets(**aux_dataset_config, split_ratio=0.05)
                aux_trainset = ConcatDataset(train_sets)
                aux_valset = ConcatDataset(val_sets)
                torch.save(aux_trainset, aux_trainset_path)
                torch.save(aux_valset, aux_valset_path)
        else:
            aux_trainset = ConcatDataset(self.test_datasets)
            aux_valset = ConcatDataset(self.val_datasets)

        # Prior knowledge datasets
        self.auxiliary_train_dataloader = DataLoader(
            aux_trainset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=self.g
        )
        self.auxiliary_val_dataloader = DataLoader(
            aux_valset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=self.g
        )
        config["auxiliary_train_dataset_size"] = len(self.auxiliary_train_dataloader.dataset)
        config["auxiliary_val_dataset_size"] = len(self.auxiliary_val_dataloader.dataset)

        assert (
            config["batch_size"] % config["inversion_batch_size"] == 0
        ), "Batch size should be divisible by inversion batch size"

        self._update_config(config)

        if config["verbose"]:
            print("Loaded auxiliary dataset with", len(self.aux_gi_t_dataset), "samples")
            print("Sample size:", self.aux_gi_t_dataset[0][0].shape, self.aux_gi_t_dataset[0][1].shape)
            print("Model / gradient size:", config["model_size"])
            print("Length of dataloader:", len(self.aux_gi_t_dataloader))

    def attack_batch(self, model, config, batch_number, original_dy_dx, dummy_inputs, dummy_targets, batch_inputs, batch_targets):
        if "num_learn_epochs" in config and config["num_learn_epochs"] <= 0:
            return None

        _, aux_gi_t_dataloader = create_gradient_inversion_dataloader(
            self.auxiliary_train_dataloader, model, config, batch_number, dummy_inputs, dummy_targets, seed_generator=self.g
        )
        _, aux_gi_v_dataloader = create_gradient_inversion_dataloader(
            self.auxiliary_val_dataloader, model, config, batch_number, dummy_inputs, dummy_targets
        )

        inversion_model, _, _ = self.initialize_inversion_model(config, dummy_inputs, dummy_targets)

        grad_to_input_optimizer, lr_schedular = self.set_attack_optimizer_and_schedular(
            inversion_model.parameters(),
            config["learn_optimizer"],
            config["learn_learning_rate"],
            config["learn_lr_decay"],
            config["num_learn_epochs"],
        )

        def generate_model_path(config, folder_path, batch_number, model, aux_gi_t_dataloader):
            keys = [
                "inversion_model",
                "defense_name",
                "attack_batch_size",
                "attack_hidden_size",
                "model_size",
                "learn_optimizer",
                "learn_learning_rate",
                "learn_lr_decay",
                "num_learn_epochs",
                "attack_loss",
                "dataset",
                "input_size",
                "output_size",
                "attack_targets",
                "seed",
                "batch_size",
                "inversion_batch_size",
                "quantiles",
            ]

            path_components = [
                folder_path,
                "grad_inputs_targets_model",
                str(batch_number),
                *(
                    str("-".join(map(str, config[key])) if isinstance(config[key], list) else config[key])
                    for key in keys
                    if key in config and config[key]
                ),
                model.name,
                "-".join(map(str, model.features)),
                str(len(aux_gi_t_dataloader.dataset)),
            ]

            return "_".join(path_components) + ".pt"

        folder_path = "../data/_model_dataset_gradients/"
        model_path = generate_model_path(config, folder_path, batch_number, model, aux_gi_t_dataloader)
        if os.path.exists(model_path) and config["load_lti_model"]:
            inversion_model.load_state_dict(torch.load(model_path))
            print("Loaded inversion model to file.")
        else:
            for epoch in range(0, config["num_learn_epochs"] + 1):
                epoch_t_loss = self.inversion_model_epoch(
                    config, epoch, aux_gi_t_dataloader, inversion_model, grad_to_input_optimizer, lr_schedular
                )
                epoch_v_loss = self.inversion_model_epoch(config, epoch, aux_gi_v_dataloader, inversion_model)

                attack_metrics = {"epoch": epoch, "aux_t_loss": np.mean(epoch_t_loss), "aux_v_loss": np.mean(epoch_v_loss)}

                self.schedular_step(config["learn_lr_decay"], lr_schedular, attack_metrics, np.mean(epoch_v_loss))

                self.evaluate_dummy_prediction(
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
            torch.save(inversion_model.state_dict(), model_path)
            print("Saved inversion model to file.")

        return inversion_model

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
        inversion_model.eval()
        with torch.no_grad():
            flattened_original_dy_dx = torch.cat([g.view(-1) for g in original_dy_dx]).to(config["device"])
            dummy_inputs, predicted_dummy_targets = inversion_model.inference(flattened_original_dy_dx.unsqueeze(0))
            # repeat dummy input along first dim until it matches batch_inputs
            dummy_inputs = dummy_inputs.repeat(batch_inputs.size(0) // dummy_inputs.size(1), 1, 1, 1)
            dummy_inputs = dummy_inputs.view(batch_inputs.size())
            if predicted_dummy_targets is not None:
                predicted_dummy_targets = predicted_dummy_targets.repeat(
                    batch_targets.size(0) // predicted_dummy_targets.size(1), 1, 1
                )
                dummy_targets = predicted_dummy_targets.view(batch_targets.size())
            self.evaluate_and_log_reconstruction(
                config,
                batch_inputs,
                batch_targets,
                dummy_inputs,
                dummy_targets,
                batch_number,
                epoch,
                config["num_learn_epochs"],
                attack_metrics,
                log_plots_n_times=10,
            )

    def initialize_inversion_model(self, config, batch_inputs, batch_targets):
        data_observations_shape = torch.Size([config["inversion_batch_size"]] + list(batch_inputs.shape[1:]))
        data_targets_shape = None
        if config["attack_targets"]:
            data_targets_shape = torch.Size([config["inversion_batch_size"]] + list(batch_targets.shape[1:]))

        inversion_model = None
        if config["inversion_model"] == "GradToInputNN":
            inversion_model = GradToInputNN(
                config["attack_hidden_size"], config["model_size"], data_observations_shape, data_targets_shape
            ).to(config["device"])
        if config["inversion_model"] == "ImprovedGradToInputNN":
            inversion_model = ImprovedGradToInputNN(
                config["attack_hidden_size"], config["model_size"], data_observations_shape, data_targets_shape
            ).to(config["device"])
        if config["inversion_model"] == "ImprovedGradToInputNN_2":
            inversion_model = ImprovedGradToInputNN_2(
                config["attack_hidden_size"], config["model_size"], data_observations_shape, data_targets_shape
            ).to(config["device"])
        return inversion_model, data_observations_shape, data_targets_shape

    def calculate_inversion_model_loss(
        self, inversion_model, config, a_batch_size, predicted_inputs, predicted_targets, aux_inputs, aux_targets
    ):
        # Flatten the inputs and targets to calculate the loss
        loss = torch.tensor(0.0).to(config["device"])
        if config["attack_loss"] == "mse":
            # View the predicted inputs and auxiliary inputs as flat vectors
            predicted_inputs = predicted_inputs.view(a_batch_size, config["inversion_batch_size"], -1)
            predicted_inputs = predicted_inputs.repeat(1, config["batch_size"] // config["inversion_batch_size"], 1)
            aux_inputs = aux_inputs.view(a_batch_size, config["batch_size"], -1)

            # Check if targets are provided and concatenate them with inputs if they are
            if predicted_targets is not None:
                predicted_targets = predicted_targets.view(a_batch_size, config["inversion_batch_size"], -1)
                predicted_targets = predicted_targets.repeat(1, config["batch_size"] // config["inversion_batch_size"], 1)

                aux_targets = aux_targets.view(a_batch_size, config["batch_size"], -1)

                # Concatenate inputs with targets along the last dimension
                predicted_combined = torch.cat((predicted_inputs, predicted_targets), dim=-1)
                aux_combined = torch.cat((aux_inputs, aux_targets), dim=-1)
            else:
                # Use only inputs if no targets are provided
                predicted_combined = predicted_inputs
                aux_combined = aux_inputs

            # Calculate pairwise squared Euclidean distances between combined vectors
            batch_wise_combined_loss = (torch.cdist(predicted_combined, aux_combined) ** 2) / predicted_combined.size(-1)

            # Solve the optimal assignment problem for the combined loss matrix
            for combined_loss_matrix in batch_wise_combined_loss:
                row_ind, col_ind = linear_sum_assignment(combined_loss_matrix.detach().cpu().numpy())
                loss += combined_loss_matrix[row_ind, col_ind].mean()

            # Normalize the loss by the batch size
            loss /= a_batch_size

        return loss

    def inversion_model_epoch(
        self, config, epoch, aux_gi_dataloader, inversion_model, grad_to_input_optimizer=None, lr_schedular=None
    ):
        epoch_loss = []
        for i, (aux_grads, aux_inputs, aux_targets) in enumerate(aux_gi_dataloader):
            if grad_to_input_optimizer is not None:
                grad_to_input_optimizer.zero_grad()
                inversion_model.train()
            else:
                inversion_model.eval()

            aux_grads, aux_inputs, aux_targets = (
                aux_grads.to(config["device"]),
                aux_inputs.to(config["device"]),
                aux_targets.to(config["device"]),
            )
            batch_size = int(aux_grads.size(0) / config["batch_size"])
            batch_num = batch_size * config["batch_size"]
            if batch_num != aux_grads.size(0):
                continue

            # print(batch_size, batch_num, aux_grads.shape, aux_inputs.shape, aux_targets.shape)
            aux_grads, aux_inputs, aux_targets = aux_grads[:batch_num], aux_inputs[:batch_num], aux_targets[:batch_num]
            aux_grads = aux_grads.view(batch_size, config["batch_size"], aux_grads.shape[-1]).mean(
                1
            )  # gradients are always averaged over batch size
            aux_inputs = aux_inputs.view(batch_size, config["batch_size"], *aux_inputs.shape[-2:])
            aux_targets = aux_targets.view(batch_size, config["batch_size"], *aux_targets.shape[-1:])  # Only 1 feature

            if aux_inputs.min() < 0 or aux_inputs.max() > 1:
                print("Aux inputs out of range:", aux_inputs.min(), aux_inputs.max())
            if aux_targets.min() < 0 or aux_targets.max() > 1:
                print("Aux targets out of range:", aux_targets.min(), aux_targets.max())

            predicted_inputs, predicted_targets = inversion_model(aux_grads)

            loss = self.calculate_inversion_model_loss(
                inversion_model, config, batch_size, predicted_inputs, predicted_targets, aux_inputs, aux_targets
            )

            if grad_to_input_optimizer is not None:
                loss.backward()
                grad_to_input_optimizer.step()

            epoch_loss.append(loss.detach().item())
            if config["verbose"]:
                if grad_to_input_optimizer is None:
                    print(f'\rEpoch {epoch}/{config["num_attack_steps"]}: Val Loss: {round(np.mean(epoch_loss), 5)}', end="")
                else:
                    print(
                        f'\rEpoch {epoch}/{config["num_attack_steps"]}: Train Loss: {round(np.mean(epoch_loss), 5)} lr: {lr_schedular.get_last_lr()}',
                        end="",
                    )
        if config["verbose"]:
            print()

        return epoch_loss


def create_gradient_inversion_dataloader(
    aux_dataloader, model, config, batch_number, dummy_inputs, dummy_targets, seed_generator=None
):
    model.to(config["device"])

    folder_path = "../data/_model_dataset_gradients/"
    # Path where the dataset will be saved or loaded from
    dataset_path = f"{folder_path}grad_inputs_targets_dataset_{config['defense_name']}_{batch_number}_{model.name}_{'-'.join(map(str, model.features))}_{config['dataset']}_{len(aux_dataloader.dataset)}_{config['input_size']}_{config['output_size']}_{config['seed']}_{config['inversion_batch_size']}.pt"

    # Check if the dataset file exists and load file
    if os.path.exists(dataset_path):
        print("Loaded gradient to inputs targets dataset:", dataset_path)
        grad_inputs_targets_dataset = torch.load(dataset_path, weights_only=False)
        config["loaded_grad_to_inputs_targets_dataset_from_file"] = True
    else:
        # If the file does not exist, proceed with creating the dataset
        config["loaded_grad_to_inputs_targets_dataset_from_file"] = False
        aux_dy_dx_inputs, aux_inputs_targets, aux_targets_targets = [], [], []
        for i, (aux_batch_inputs, aux_batch_targets) in enumerate(aux_dataloader):
            aux_batch_inputs, aux_batch_targets = (
                aux_batch_inputs[:, :, model.features].to(config["device"]),
                aux_batch_targets[:, :, 0].to(config["device"]),
            )

            if aux_batch_inputs.shape[1] != dummy_inputs.shape[1]:
                aux_batch_inputs = interpolate(aux_batch_inputs, dummy_inputs.shape[1])
            if aux_batch_targets.shape[1] != dummy_targets.shape[1]:
                aux_batch_targets = interpolate(aux_batch_targets.unsqueeze(-1), dummy_targets.shape[1]).squeeze(-1)

            model.zero_grad()
            aux_out = model(aux_batch_inputs)
            aux_y = F.mse_loss(aux_out, aux_batch_targets)
            aux_y.backward()

            if "defense_name" in config:
                # Apply gradient defenses
                gradients = [param.grad for param in model.parameters()]
                if "sign" in config:
                    gradients = apply_sign_transformation(gradients)
                if "prune_rate" in config:
                    gradients = apply_pruning(gradients, config["prune_rate"])
                if "dp_epsilon" in config:
                    gradients = add_gaussian_noise(gradients, config["dp_epsilon"])

                # Update model parameters with modified gradients
                for param, grad in zip(model.parameters(), gradients):
                    param.grad = grad

            flattened_aux_dy_dx = torch.cat([p.grad.detach().view(-1) for p in model.parameters()]).unsqueeze(
                0
            )  # add batch dimension but is always 1

            # flattend aux_dy_dx should be of shape (batch_size, model_size)
            aux_dy_dx_inputs.append(flattened_aux_dy_dx.clone().detach().cpu())
            aux_inputs_targets.append(aux_batch_inputs.clone().detach().cpu())
            aux_targets_targets.append(aux_batch_targets.clone().detach().cpu())

        aux_dy_dx_inputs, aux_inputs_targets, aux_targets_targets = (
            torch.stack(aux_dy_dx_inputs),
            torch.stack(aux_inputs_targets),
            torch.stack(aux_targets_targets),
        )
        grad_inputs_targets_dataset = TensorDataset(aux_dy_dx_inputs, aux_inputs_targets, aux_targets_targets)

        # Save the dataset to file
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(grad_inputs_targets_dataset, dataset_path)

    if seed_generator is not None:
        return grad_inputs_targets_dataset, DataLoader(
            grad_inputs_targets_dataset,
            batch_size=config["attack_batch_size"] * config["batch_size"],
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=seed_generator,
        )
    return grad_inputs_targets_dataset, DataLoader(
        grad_inputs_targets_dataset, batch_size=config["attack_batch_size"] * config["batch_size"]
    )
