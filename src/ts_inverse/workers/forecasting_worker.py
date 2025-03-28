from copy import deepcopy
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
import time

from ts_inverse.utils import set_seed, seed_worker
from .worker import Worker
from ts_inverse.datahandler import ConcatSliceDataset


class ForecastingWorker(Worker):
    def __init__(self, worker_id):
        worker_name = "ForecastingWorker"
        super().__init__(worker_id, worker_name)

    def worker_process(self, c, d_c, m_c, fam_c):
        final_model_settings = {**m_c, **fam_c}  # Merge specific model and for all model settings
        train_datasets, val_datasets, test_datasets = self.get_datasets(**d_c)
        freq_in_day = train_datasets[0].freq_in_day
        model_settings = {key: value for key, value in final_model_settings.items() if not key.startswith("_")}
        for key, value in model_settings.items():
            if key.startswith("input_") or key.startswith("output_"):
                model_settings[key] = value * freq_in_day
                final_model_settings[key] = value * freq_in_day

        model = final_model_settings["_model"](**model_settings)  # Create model
        final_model_settings.update(model.extra_info)  # Merge extra info from model
        final_config = {**c, **d_c, **final_model_settings}  # Merge all settings
        del final_config["_model"]
        final_config["model_name"] = model.name
        final_config["model"] = model.name

        print("Starting training", final_config["run_number"], "with config:", final_config)
        self.train_model(model, train_datasets, final_config, val_datasets, test_datasets)

    def init_logger_object(self, config):
        tags = ["Forecasting"]
        project_names = {
            "wandb": "ts-inverse_forecasting",
        }

        self._init_logger_object(project_names, tags, config)

    def train_model(self, model, train_datasets, config, val_datasets, test_datasets):
        g = set_seed(config["seed"])  # Set seed
        tr_dataloader = DataLoader(
            ConcatSliceDataset(train_datasets),
            batch_size=config["batch_size"],
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )  # Create dataloader

        self.init_logger_object(config)
        val_dataset = ConcatSliceDataset(val_datasets)
        test_dataset = ConcatSliceDataset(test_datasets)
        learning_rate = config["learning_rate"]
        num_epochs = config["num_epochs"]
        device = config["device"]
        verbose = config["verbose"]
        early_stopping_patience = config["early_stopping_patience"]
        early_stopping_min_delta = config["early_stopping_min_delta"]
        best_val_loss = np.inf
        best_model_state_dict = model.state_dict()
        epochs_without_improvement = 0
        early_stop = False
        total_gradient_updates = 0

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        model.to(device)
        model.train()
        start_time = time.time()
        for epoch in range(1, num_epochs + 1):
            if early_stop:
                break

            epoch_metrics = {"epoch": epoch}
            epoch_losses = []
            for inputs, targets in tr_dataloader:
                inputs, targets = inputs[:, :, model.features].to(device), targets[:, :, 0].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                mse_loss = F.mse_loss(outputs, targets)
                mse_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    rmse_loss = torch.sqrt(mse_loss)
                    mae_loss = F.l1_loss(outputs, targets)
                    epoch_losses.append(
                        {
                            "train/loss_mse": mse_loss.detach().item(),
                            "train/loss_rmse": rmse_loss.detach().item(),
                            "train/loss_mae": mae_loss.detach().item(),
                        }
                    )
                total_gradient_updates += 1

            epoch_metrics["train/total_gradient_updates"] = total_gradient_updates
            mean_epoch_losses = {k: np.mean([loss[k] for loss in epoch_losses]) for k in epoch_losses[0].keys()}
            epoch_metrics.update(mean_epoch_losses)

            ## Evaluate model on validation set
            val_losses = evaluate_model(model, val_dataset, device, "validation")
            epoch_metrics.update(val_losses)
            current_val_loss = val_losses["validation/loss_mse"]
            if current_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = current_val_loss
                best_model_state_dict = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                early_stop = True

            if epoch % max(10, (num_epochs // 10)) == 0 or early_stop:
                # Log weekly load profile plot
                for val_set in val_datasets:
                    consumer_id = val_set.name.split("_")[0]
                    fig, _ = val_set.plot_weekly_load_profile(
                        median_instead_of_average=False, model=model, device=device, it=True
                    )
                    self._log_matplotlib_figure(fig, step=epoch, log_name=f"_val_{consumer_id}", matplotlib_only=True)

            if verbose:
                print(
                    f"Epoch {epoch}/{num_epochs}, Loss: {mean_epoch_losses['train/loss_mse']}, Val Loss: {current_val_loss}, Best Val Loss: {best_val_loss}"
                )

            self._log_metrics(epoch_metrics, step=epoch)

        if verbose:
            print(f"Total Training duration: {time.time() - start_time:.2f} seconds")

        ### Evaluate model on test set
        model.load_state_dict(best_model_state_dict)
        test_losses = evaluate_model(model, test_dataset, device, "test")
        self._log_metrics(test_losses, step=num_epochs + 1)

        ### Plot the best validation model on all datasets
        for train_set in train_datasets:
            consumer_id = train_set.name.split("_")[0]
            fig, _ = train_set.plot_weekly_load_profile(median_instead_of_average=False, model=model, device=device, it=True)
            self._log_matplotlib_figure(fig, step=num_epochs + 1, log_name=f"_train_{consumer_id}", matplotlib_only=True)

        for val_set in val_datasets:
            consumer_id = val_set.name.split("_")[0]
            fig, _ = val_set.plot_weekly_load_profile(median_instead_of_average=False, model=model, device=device, it=True)
            self._log_matplotlib_figure(fig, step=epoch, log_name=f"_val_{consumer_id}", matplotlib_only=True)

        for test_set in test_datasets:
            consumer_id = test_set.name.split("_")[0]
            fig, _ = test_set.plot_weekly_load_profile(median_instead_of_average=False, model=model, device=device, it=True)
            self._log_matplotlib_figure(fig, step=num_epochs + 1, log_name=f"_test_{consumer_id}", matplotlib_only=True)

        artifact = wandb.Artifact("final_model", type="model")
        torch.save(best_model_state_dict, os.path.join(wandb.run.dir, "final_model.pth"))
        artifact.add_file(os.path.join(wandb.run.dir, "final_model.pth"))
        wandb.log_artifact(artifact)

        if verbose:
            print(f"Test Loss: {test_losses}")

        model.eval()
        wandb.finish()
        return time.time() - start_time


def evaluate_model(model, dataset, device="cpu", name="validation"):
    model.to(device)
    model.eval()
    with torch.no_grad():
        inputs, targets = dataset[:]
        inputs, targets = inputs[:, :, model.features].to(device), targets[:, :, 0].to(device)
        outputs = model(inputs)
        mse_loss = F.mse_loss(outputs, targets)
        rmse_loss = torch.sqrt(mse_loss)
        mae_loss = F.l1_loss(outputs, targets)
        del inputs, targets  # delete tensors to free up memory.

    model.train()
    losses = {
        f"{name}/loss_mse": mse_loss.detach().item(),
        f"{name}/loss_rmse": rmse_loss.detach().item(),
        f"{name}/loss_mae": mae_loss.detach().item(),
    }

    return losses
