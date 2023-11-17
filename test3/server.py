from time import sleep
import flwr as fl
import numpy as np
from flwr.common import (
    FitIns,
    Parameters,
    ndarrays_to_parameters
)
from utils import randomprefix
import requests

url = "http://167.71.139.232:12314/api/secure-aggregation/job-id/"

triggerBody = {
    "computationType": "fsum",
    "returnUrl": "http://localhost:4100",
    "clients": ["ZuellingPharma", "ChildrensHospital"]
}

resultUrl = "http://167.71.139.232:12314/api/get-result/job-id/"


class CustomFed(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(min_available_clients=2)

    def aggregate_fit(self,
                      server_round,
                      results,
                      failures):

        response = requests.post(
            url + "testKey" + randomprefix + str(server_round), json=triggerBody)
        if response.ok:
            print("Request was successful!")
            print(response.text)
        else:
            print(f"Request failed with status code {response.status_code}.")
            print(response.text)

        while 1:
            response = requests.get(
                resultUrl + "testKey" + randomprefix + str(server_round))
            print("Response got ", resultUrl + "testKey" + randomprefix +
                  str(server_round), response)
            if response.ok:
                print("Request was successful!")
                json_data = response.json()
                print("Result:", json_data)
                if "computationOutput" in json_data:
                    print("Result:", json_data["computationOutput"])
                    first = np.array(
                        json_data["computationOutput"][:-10]).reshape(-1, 10)
                    second = np.array(json_data["computationOutput"][-10:])
                    print("In here", first, second)
                    print("results", results)
                    res = ndarrays_to_parameters([first, second])
                    print("FINAL RESULT", res)
                    return super().aggregate_fit(server_round, results, failures)
            else:
                print(
                    f"Request failed with status code {response.status_code}.")
                print(response.text)
            sleep(1)
        return super().aggregate_fit(server_round, results, failures)

    def configure_fit(
        self, server_round: int, parameters, client_manager
    ):
        """Configure the next round of training."""
        config = {"round": server_round}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]


strategy = CustomFed()
fl.server.start_server(config=fl.server.ServerConfig(
    num_rounds=3), strategy=strategy)
