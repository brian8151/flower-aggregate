import flwr as fl
import sys
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.fedavg import FedAvg
from logging import INFO
from flwr.common.logger import log
import numpy as np

class OnyxCustomStrategy(FedAvg):


    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)

        for client, fit_ins in fit_ins_list:
            client_properties = client.properties
            #print(f"OnyxCustomStrategy [Server] Client ID: {client.cid}, Properties: {client_properties}")
            log(INFO,"OnyxCustomStrategy Client ID (%s) Properties (%s).", client.cid,client_properties)
        return fit_ins_list

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        evaluate_ins_list = super().configure_evaluate(server_round, parameters, client_manager)

        # Request properties from clients
        for client, evaluate_ins in evaluate_ins_list:
            client_properties = client.properties
            print(f"OnyxCustomStrategy [Server] Client ID: {client.cid}, Properties: {client_properties}")

        return evaluate_ins_list

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
                print("Aggregated parameters:")
                print(aggregated_parameters)
                log(INFO, " -------> Saving round (%s) aggregated_ndarrays... <-----", server_round)
                # Save aggregated_ndarrays
                np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)
            return aggregated_parameters, aggregated_metrics