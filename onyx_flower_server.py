import logging
from typing import Optional, Tuple, Dict

from flwr.server.server import Server, EvaluateResultsAndFailures, FitResultsAndFailures
from flwr.server.history import History
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy, FedAvg
from flwr.common import Parameters, Scalar
from logging import INFO
from flwr.common.logger import log

class OnyxFlowerServer(Server):
    """Custom Flower server with additional logging."""

    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        log(INFO, "OnyxFlowerServer initialized.")

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """Run federated learning for a number of rounds."""
        log(INFO, "Starting fit for (%s) rounds with timeout %s", num_rounds, timeout)
        result = super().fit(num_rounds, timeout)
        log(INFO, "Fit completed")
        return result

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Evaluate model."""
        log(INFO, "Starting evaluation for (%s) rounds with timeout %s", server_round, timeout)
        result = super().evaluate_round(server_round, timeout)
        log(INFO, "Evaluation round completed")
        return result

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Fit model for a single round."""
        log(INFO, "Starting fit round for (%s) rounds with timeout %s", server_round, timeout)
        result = super().fit_round(server_round, timeout)
        log(INFO, "Fit round completed")
        return result

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Disconnect all clients."""
        super().disconnect_all_clients(timeout)
        log(INFO, "All clients disconnected")