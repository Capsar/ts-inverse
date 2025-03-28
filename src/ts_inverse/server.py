import argparse
import flwr as fl
import warnings
from flwr.common import parameters_to_ndarrays
from ts_inverse import models, utils

warnings.filterwarnings("ignore")


class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        weights_results = [(client_proxy.cid, parameters_to_ndarrays(fit_res.parameters)) for client_proxy, fit_res in results]
        for client_id, weights in weights_results:
            print(f"Client {client_id} weights shapes: {[weight.shape for weight in weights]}")

        weights = super().aggregate_fit(rnd, results, failures)
        # if weights is not None:
        #     # Save weights
        #     print(f"Saving round {rnd} weights...")
        #     np.savez(f"../out/04_federated_learning_models/round-{rnd}-weights.npz", *weights)
        return weights


def evaluate_metrics_aggregation(metrics):
    print("Evaluate Metrics:", metrics)
    # Multiply accuracy of each client by number of examples used
    maes = [num_examples * m["mae"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"mae": sum(maes) / sum(examples)}


def fit_config(server_round):
    return {"epochs": 1, "lr": 0.01, "batch_size": 1}


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use only 10 datasamples for validation. \
            Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GRU_Predictor",
        choices=models.model_classes.keys(),
        required=False,
        help=f"Specifies the model to be used: {models.model_classes.keys()}",
    )

    args = parser.parse_args()

    # Initialize model weights
    model = models.initialize_model_by_name(args.model)
    model_parameters = utils.get_model_parameters(model)

    print(f"Loaded {model.name} parameters!")

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        fit_config=fit_config,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
