
from flwr.server.superlink.fleet.grpc_bidi.grpc_client_proxy  import GrpcClientProxy
from flwr.server.superlink.fleet.grpc_bidi.grpc_bridge import (GrpcBridge)
from logging import INFO, basicConfig, getLogger
from typing import Optional
from flwr import common

class OnyxGrpcClientProxy(GrpcClientProxy):
    """Custom gRPC client proxy with additional logging."""

    def __init__(self, cid: str, bridge: GrpcBridge):
        super().__init__(cid, bridge)
        self.logger = getLogger(f"OnyxGrpcClientProxy-{cid}")
        basicConfig(level=INFO)
        self.logger.info(f"OnyxGrpcClientProxy initialized with client ID: {cid}")

    def fit(self, ins: common.FitIns, timeout: Optional[float], group_id: Optional[int]) -> common.FitRes:
        """Refine the provided parameters using the locally held dataset."""
        self.logger.info(f"Onyx Client {self.cid}: fit method called with timeout {timeout} and group_id {group_id}.")
        result = super().fit(ins, timeout, group_id)
        self.logger.info(f"Onyx Client {self.cid}: fit method completed with result {result}.")
        return result

    def evaluate(self, ins: common.EvaluateIns, timeout: Optional[float], group_id: Optional[int]) -> common.EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""
        self.logger.info(f"Onyx Client {self.cid}: evaluate method called with timeout {timeout} and group_id {group_id}.")
        result = super().evaluate(ins, timeout, group_id)
        self.logger.info(f"Onyx Client {self.cid}: evaluate method completed with result {result}.")
        return result

    def __repr__(self):
        """Represent the client proxy for logging purposes."""
        return f"OnyxGrpcClientProxy(client_id={self.cid})"