import time
import flwr as fl
import numpy as np

from collections import namedtuple

# Define FitIns as a namedtuple with parameters and config
FitIns = namedtuple("FitIns", ["parameters", "config"])

# Define EvaluateIns as a namedtuple with parameters and config
EvaluateIns = namedtuple("EvaluateIns", ["parameters", "config"])


class CustomStrategy(fl.server.strategy.Strategy):

    def __init__(self, min_num_clients: int = 2):
        self.min_num_clients = min_num_clients

    def initialize_parameters(self, client_manager):

        print("Initializing parameters")

        # Initialize parameters with random values
        # This is just a placeholder. You should replace this with your actual initialization logic.
        return [np.random.randn(10, 10).tolist()]

    def configure_fit(self, server_round, parameters, client_manager):
        # Configure clients for training
        # For simplicity, we're sending the same configuration to all available clients
        num_clients = client_manager.num_available()
        while client_manager.num_available() < self.min_num_clients:
            print(
                f"Waiting for at least {self.min_num_clients} clients to be available. Currently, {client_manager.num_available()} are available.")
            time.sleep(10)

        clients = client_manager.clients
        print(f"Configuring round {server_round}")
        return [(client, FitIns(parameters, {})) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        # Aggregate training results by averaging
        # This is a basic averaging logic. You might want to replace this with your aggregation logic.

        print("aggregate_fit", results, server_round, failures)

        aggregated_parameters = [
            np.mean([res[1].parameters[0] for res in results], axis=0).tolist()]
        return aggregated_parameters, {"loss": np.mean([res[1].loss for res in results])}

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Configure clients for evaluation

        print("configure_evaluate ", server_round, parameters, client_manager)
        clients = client_manager.clients
        return [(client, EvaluateIns(parameters, {})) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        # Aggregate evaluation results by averaging
        print("aggregate_evaluate", results, server_round, failures)
        return np.mean([res[1].loss for res in results]), {"accuracy": np.mean([res[1].metrics["accuracy"] for res in results])}

    def evaluate(self, rnd, parameters):
        # Evaluate the current model parameters
        # This is a placeholder. You should replace this with your actual evaluation logic.
        print("evaluate", rnd, parameters)
        return 1.0, {"accuracy": 0.5}
