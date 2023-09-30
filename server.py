import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict, Optional, Tuple
# from custom_server_strategy import CustomStrategy

# import logging

# logging.getLogger('flwr').setLevel(logging.CRITICAL)
# logging.basicConfig(level=logging.CRITICAL)


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    print("fit_round", server_round)
    return {"server_round": server_round}


def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in
    # `evaluate` itself
    _, (X_test, y_test) = utils.load_mnist()

    # The `evaluate` function will be called after every round
    def evaluate(server_round: int, parameters, config):
        # Update model with the latest parameters
        # utils.set_model_params(model, parameters)
        print("X_test", X_test.shape, y_test.shape)

        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    # utils.set_initial_params(model)

    _, (X_test, y_test) = utils.load_mnist()
    model.fit(X_test, y_test)
    print("model", model.coef_, model.intercept_)
    get_eval_fn(model)(1, utils.get_model_parameters(model), {})
    # strategy = fl.server.strategy.FedAvg(
    #     min_available_clients=2,
    #     evaluate_fn=get_eval_fn(model),
    #     on_fit_config_fn=fit_round,
    # )
    # fl.server.start_server(server_address="0.0.0.0:8080",
    #                        strategy=strategy, config=fl.server.ServerConfig(num_rounds=3))
    # strategy = CustomStrategy()
    # fl.server.start_server(server_address="0.0.0.0:8080",
    #                        strategy=strategy, config=fl.server.ServerConfig(num_rounds=3))
