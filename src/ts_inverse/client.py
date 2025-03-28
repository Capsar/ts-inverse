from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch
import flwr as fl
from collections import OrderedDict
import warnings

import datahandler
import utils

warnings.filterwarnings("ignore")


class TimeSeriesClient(fl.client.NumPyClient):
    def __init__(self, model_config: dict, dataset_config: dict, datasets, device: str):
        trainset, valset, testset = datasets
        trainset, valset, testset = trainset[0], valset[0], testset[0]

        self.model_config = model_config
        self.model = model_config["_model"](
            features=[0],
            hidden_size=model_config["hidden_size"],
            input_size=trainset.freq_in_day * dataset_config["observation_days"],
            output_size=trainset.freq_in_day * dataset_config["future_days"],
        )
        self.device = device
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        print(f"Client {dataset_config['columns']} initialized.")

    def get_parameters(self):
        """Return the current parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Initialize random model and replace its parameters with the ones given."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        self.set_parameters(parameters)
        lr, batch_size, epochs = config["lr"], config["batch_size"], config["epochs"]
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
        )
        train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        train_loss_history = utils.train_model(self.model, optimizer, train_loader, num_epochs=epochs, device=self.device)
        results = {"train_mse_loss": train_loss_history[-1]}
        return self.get_parameters(), len(self.trainset), results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        self.set_parameters(parameters)
        test_mse_loss = utils.evaluate_model(self.model, self.testset, device=self.device)
        test_mae_loss = utils.evaluate_model(self.model, self.testset, criterion=F.l1_loss, device=self.device)
        return float(test_mse_loss), len(self.testset), {"mae": float(test_mae_loss)}


def client_factory(d_config, m_config, device, num_clients):
    n_gpus = 0
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {n_gpus}")

    print(f"Loading dataset for {num_clients} clients:", d_config)
    dataset_df, dataset_class = datahandler.get_dataset_df(d_config["dataset"])
    datasets_for_client_ids = []
    del d_config["dataset"]
    for i in range(num_clients):
        datasets_for_client_ids.append(datahandler.get_datasets_from_df(dataset_df, dataset_class, i, **d_config))
    print("Finished loading datasets.")

    def client_fn(client_id):
        print(f"Loading client {client_id}")
        d_config["columns"] = int(client_id)
        return TimeSeriesClient(m_config, d_config, datasets_for_client_ids[int(client_id)], device).to_client()

    return client_fn


# def main() -> None:
#     parser = argparse.ArgumentParser(description="Flower")
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="GRU_Predictor",
#         choices=models.model_classes.keys(),
#         required=False,
#         help=f"Specifies the model to be used: {models.model_classes.keys()}"
#     )
#     parser.add_argument(
#         "--dataset",
#         type=str,
#         default="london_smartmeter",
#         choices=['london_smartmeter', 'electricity_321', 'electricity_370', 'kddcup'],
#         required=True,
#         help="Specifies the dataset to be used."
#     )
#     parser.add_argument(
#         "--partition",
#         type=int,
#         default=0,
#         choices=range(0, 10),
#         required=False,
#         help="Specifies which consumer column to pick from the project energy data."
#     )
#     parser.add_argument(
#         "--toy",
#         type=bool,
#         default=False,
#         required=False,
#         help="Set to true to quicky run the client using only 10 datasamples. Useful for testing purposes. Default: False"
#     )
#     parser.add_argument(
#         "--use_cuda",
#         type=bool,
#         default=False,
#         required=False,
#         help="Set to true to use GPU. Default: False"
#     )

#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
#     client = client_factory(args.dataset, args.model, args.toy, device)(args.partition)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


# if __name__ == "__main__":
#     main()
