import flwr as fl
from typing import Dict, Tuple, List
import numpy as np


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model_params):  # Only need model_params
        self.model_params = model_params  # Store model_params

    def get_parameters(self, config: Dict[str, str]):
        # print(f"Returning the get parameters to the server: {self.model_params}") #for debug reasons
        return self.model_params

    def fit(self, parameters: fl.common.NDArrays, config: Dict[str, str]):
        # Return just the model_params from the local training
        # parameters: The current weights of the model
        # config: Dict of training parameters
        print("Returning the weights from training")  # for debug reasons
        return self.model_params, 0, {}  # No local training

    def evaluate(self, parameters: fl.common.NDArrays, config: Dict[str, str]):
        # No evaluation return a default
        print("Skipping evaluation")  # for debug reasons
        return 0.0, 0, {}  # loss, number of examples, metrics
