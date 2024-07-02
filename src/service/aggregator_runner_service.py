import numpy as np
from src.util import log
from src.repository.model.model_data_repositoty import get_model_client_training_record
from src.ml.model_builder import decompress_weights
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Code, FitRes, Status, EvaluateRes, EvaluateIns
from flwr.common.typing import List, Tuple, Union
from flwr.common import Metrics
from src.common.onyx_custom_client_proxy import CustomClientProxy
from src.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays

logger = log.init_logger()


class AggregatorRunner:
    """ Class for machine learning service """

    def __init__(self):
        pass

    def aggregate(self, workflow_trace_id, domain):
        try:
            training_record = get_model_client_training_record(workflow_trace_id, domain)
            if training_record:
                logger.info(f"Retrieved training record: {training_record}")
                client_id = training_record['client_id']
                weights_encoded = training_record['parameters']
                num_examples = training_record['num_examples']
                loss = training_record['loss']

                # Deserialize parameters
                weights = decompress_weights(weights_encoded)
                metrics = {"accuracy": training_record['accuracy']}
                metrics_collected = []
                weights_collected = []
                metrics_collected.append((num_examples, metrics))
                weights_collected.append((num_examples, weights))
                weights_only = [weight for _, weights in weights_collected for weight in weights]
                for idx, weight in enumerate(weights_only):
                    print(f"Weight {idx} shape: {weight.shape}, dtype: {weight.dtype}")

                agg_parameters = ndarrays_to_parameters(weights_only)
                # Aggregate metrics
                aggregated_metrics = weighted_average(metrics_collected)
                print("Aggregated Metrics:", aggregated_metrics)
                # Assuming agg_parameters are now correctly processed
                fedavg = FedAvg()
                client_proxy = CustomClientProxy(cid=client_id)
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
                print(f"fedavg.aggregate_fit --------------------->")
                parameters_aggregated, metrics_aggregated = fedavg.aggregate_fit(1, results, failures)
                print(f"check parameters_aggregated --------------------->")
                if parameters_aggregated is not None:
                    print(".......................saving parameters_aggregated.......................")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays = parameters_to_ndarrays(parameters_aggregated)
                    print("saved parameters_aggregated to db DB Model weights:", aggregated_ndarrays)
                    print(f"metrics_aggregated {metrics_aggregated}")

            else:
                logger.error(f"No training record found for workflow_trace_id {workflow_trace_id}")
        except Exception as e:
            logger.error(f"Error getting model weights with compression: {e}")
            raise



def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    logger.info(" set up weighted_average")
    # Calculate weighted accuracies
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Log each client's accuracy and the number of examples used
    for i, (acc, ex) in enumerate(zip(accuracies, examples)):
        logger.info(f"Client {i}: Accuracy={acc / ex}, Examples={ex}")

    total_examples = sum(examples)
    total_weighted_accuracy = sum(accuracies)
    weighted_avg_accuracy = total_weighted_accuracy / total_examples
    # Log aggregate information
    logger.info(f"Total Examples: {total_examples}")
    logger.info(f"Total Weighted Accuracy: {total_weighted_accuracy}")
    logger.info(f"Weighted Average Accuracy: {weighted_avg_accuracy}")

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": weighted_avg_accuracy}