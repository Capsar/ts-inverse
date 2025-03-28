import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import wandb
import os
from ts_inverse.datahandler import TimeSeriesDataSet, get_datasets


class Worker:
    def __init__(self, worker_id: int, worker_name: str):
        self.worker_id = worker_id
        self.worker_name = worker_name
        self.logger_object_dict = {}

    def __str__(self):
        return f"Worker ID: {self.worker_id}, Worker Name: {self.worker_name}"

    def worker_process(self, **kwargs):
        raise NotImplementedError("Worker process not implemented")

    def _init_logger_object(self, project_names, tags, config):
        name = f"{config['model']}_{datetime.datetime.now().strftime('%m-%d_%H%M%S')}"
        tags.extend([config["experiment_name"], config["dataset"], config["model"]])
        self.logger_object_dict = {}

        if "wandb" in config["logger_service"]:
            settings = wandb.Settings(disable_job_creation=True)
            wandb.init(
                project=project_names["wandb"],
                entity=os.getenv("WANDB_ENTITY"),
                tags=tags,
                name=name,
                config=config,
                settings=settings,
            )
            self.logger_object_dict["wandb"] = None
        if "comet_ml" in config["logger_service"]:
            from comet_ml import Experiment

            # Lacking logging of more than 10000 metrics per 45 seconds.
            experiment = Experiment(
                api_key=os.getenv("COMET_ML_API_KEY"),
                project_name=project_names["comet_ml"],
                workspace="capsar",
                log_code=False,
                log_git_metadata=False,
                log_git_patch=False,
            )
            experiment.set_name(name)
            for tag in tags:
                experiment.add_tag(tag)
            experiment.log_parameters(config)
            self.logger_object_dict["comet_ml"] = experiment
        if "neptune" in config["logger_service"]:
            import neptune

            neptune_run = neptune.init_run(
                project=project_names["comet_ml"],
                api_token=os.getenv("NEPTUNE_API_TOKEN"),
                name=name,
                tags=[tags],
                capture_hardware_metrics=False,
                source_files=[],
            )
            neptune_run["parameters"] = config
            self.logger_object_dict["neptune"] = neptune_run
        if "clear_ml" in config["logger_service"]:
            from clearml import Task

            task = Task.init(project_name=project_names["clear_ml"], task_name=name)
            task.set_parameters(config)
            task.add_tags(tags)
            self.logger_object_dict["clear_ml"] = task

    def _update_config(self, config):
        if "wandb" in self.logger_object_dict.keys():
            run_config = wandb.run.config
            run_config.update(config)

    def _log_dataframe(self, df, step, log_name=""):
        if "wandb" in self.logger_object_dict.keys():
            wandb.log({f"dataframe{log_name}": wandb.Table(dataframe=df), "custom_step": step})
        if "comet_ml" in self.logger_object_dict.keys():
            self.logger_object_dict["comet_ml"].log_dataframe(df, step=step)
        if "neptune" in self.logger_object_dict.keys():
            from neptune.types import File

            self.logger_object_dict["neptune"][f"attack/dataframe{log_name}"].upload(
                File.as_html(df), name=f"dataframe{log_name}", step=step
            )
        if "clear_ml" in self.logger_object_dict.keys():
            self.logger_object_dict["clear_ml"].get_logger().report_table(
                f"attack/dataframe{log_name}", series="series", table_plot=df, iteration=step
            )

    def _log_matplotlib_figure(self, fig, step, log_name="", matplotlib_only=False):
        if "wandb" in self.logger_object_dict.keys():
            if matplotlib_only:
                wandb.log({f"figure_matplotlib{log_name}": wandb.Image(fig), "custom_step": step})
            else:
                wandb.log(
                    {f"figure_interactive{log_name}": fig, f"figure_matplotlib{log_name}": wandb.Image(fig), "custom_step": step}
                )
        if "comet_ml" in self.logger_object_dict.keys():
            self.logger_object_dict["comet_ml"].log_figure(fig, step=step)
        if "neptune" in self.logger_object_dict.keys():
            from neptune.types import File

            self.logger_object_dict["neptune"][f"attack/figure{log_name}"].upload(
                File.as_image(fig), name=f"figure{log_name}", step=step
            )
        if "clear_ml" in self.logger_object_dict.keys():
            self.logger_object_dict["clear_ml"].get_logger().report_matplotlib_figure(
                f"attack/figure{log_name}", series="series", figure=fig, iteration=step
            )
        plt.close(fig)

    def _log_metrics(self, metrics_dict, step):
        if "wandb" in self.logger_object_dict.keys():
            wandb.log({**metrics_dict, "custom_step": step})
        if "comet_ml" in self.logger_object_dict.keys():
            self.logger_object_dict["comet_ml"].log_metrics(metrics_dict, step=step)
        if "neptune" in self.logger_object_dict.keys():
            self.logger_object_dict["neptune"]["attack"].append(metrics_dict, step=step)
        if "clear_ml" in self.logger_object_dict.keys():
            for key, value in metrics_dict.items():
                name_spaces = key.split("/")
                if len(name_spaces) == 3:
                    if isinstance(value, (int, float)):
                        self.logger_object_dict["clear_ml"].get_logger().report_scalar(
                            f"{name_spaces[0]}_{name_spaces[1]}", series=f"{name_spaces[2]}", value=value, iteration=step
                        )
                else:
                    if isinstance(value, (int, float)):
                        self.logger_object_dict["clear_ml"].get_logger().report_scalar(
                            f"{key}", series="series", value=value, iteration=step
                        )
                    elif isinstance(value, (list, np.ndarray)):
                        self.logger_object_dict["clear_ml"].get_logger().report_table(
                            f"{key}", series="series", table_plot=pd.DataFrame(value), iteration=step
                        )
                    else:
                        print("Unknown type: for key", key, ":", type(value), ":", value)

    def _end_logger_object(self):
        if "wandb" in self.logger_object_dict.keys():
            wandb.finish()
        if "comet_ml" in self.logger_object_dict.keys():
            self.logger_object_dict["comet_ml"].end()
        if "neptune" in self.logger_object_dict.keys():
            self.logger_object_dict["neptune"].stop()
        if "clear_ml" in self.logger_object_dict.keys():
            self.logger_object_dict["clear_ml"].get_logger()
            self.logger_object_dict["clear_ml"].close()

    def get_datasets(
        self,
        dataset,
        normalize,
        columns,
        train_stride,
        observation_days,
        future_days,
        validation_stride=24,
        split_ratio=0.2,
        should_dropna=True,
    ) -> tuple[list[TimeSeriesDataSet], list[TimeSeriesDataSet], list[TimeSeriesDataSet]]:
        return get_datasets(
            dataset,
            normalize,
            columns,
            train_stride,
            observation_days,
            future_days,
            validation_stride,
            split_ratio,
            should_dropna,
        )
