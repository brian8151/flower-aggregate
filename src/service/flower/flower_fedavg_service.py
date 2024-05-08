from src.common.flower.onyx_custom_flwr_client_proxy import CustomFlowerClientProxy
from src.model.message_req import MessageRequest
from src.strategy.flower.flower_fedavg import OnyxFlowerFedAvgStrategy
from flwr.common import Code, FitRes, Status, EvaluateRes, EvaluateIns
from flwr.common.typing import List, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from src.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from src.util import log
logger = log.init_logger()

class FlowerFedAvgService:
    def __init__(self):
        pass

    def run_aggregate_fit(self, message: MessageRequest):
        logger.info(f"Processing FedAvg: client_id: {message.client_id}, message_id: {message.message_id}")
        # Deserialize the parameters to model weights if needed
        agg_parameters = parameters_to_ndarrays(message.parameters)
        num_examples = message.num_examples
        metrics_collected = message.metrics
        fedavg = OnyxFlowerFedAvgStrategy()
        client_proxy = CustomFlowerClientProxy(cid=message.client_id)
        results: List[Tuple[ClientProxy, FitRes]] = [
            (
                client_proxy,
                FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=agg_parameters,
                    num_examples=num_examples,
                    metrics=metrics_collected,
                ),
            )
        ]
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
        logger.info("fedavg.aggregate_fit --------------------->")
        parameters_aggregated, metrics_aggregated = fedavg.aggregate_fit(1, results, failures)
        logger.info("check parameters_aggregated --------------------->")
        if parameters_aggregated is not None:
            logger.info(".......................saving parameters_aggregated.......................")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays = parameters_to_ndarrays(parameters_aggregated)
            logger.info("saved parameters_aggregated to db DB Model weights: {0}".format(aggregated_ndarrays))
            logger.info("metrics_aggregated {0}".format(metrics_aggregated))

        logger.info("Applying FedAvg to Flower strategy.")

