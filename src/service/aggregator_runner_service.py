import numpy as np
from src.util import log
from src.repository.model.model_data_repositoty import get_model_client_training_record
from src.ml.model_builder import decompress_weights, compress_weights
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

                # Decompress weights
                decompressed_weights = decompress_weights(weights_encoded)
                ser_parameters = ndarrays_to_parameters(decompressed_weights)
                weights_as_ndarrays = parameters_to_ndarrays(ser_parameters)

                metrics = {"accuracy": training_record['accuracy']}
                metrics_collected = []
                weights_collected = []
                metrics_collected.append((num_examples, metrics))
                weights_collected.append((num_examples, weights_as_ndarrays))

                logger.info(f"metrics_collected: {metrics_collected}")
                weights_only = [weight for _, weights in weights_collected for weight in weights]

                logger.info(f"loop weights_only")
                for idx, weight in enumerate(weights_only):
                    logger.info(f"Weight {idx} shape: {weight.shape}, dtype: {weight.dtype}")

                logger.info(f"weights_only ndarrays_to_parameters")
                agg_parameters = ndarrays_to_parameters(weights_only)

                # Aggregate metrics
                logger.info(f"weighted_average")
                aggregated_metrics = weighted_average(metrics_collected)
                logger.info(f"Aggregated Metrics: {aggregated_metrics}")

                fedavg = FedAvg(
                    fraction_fit=0.2,
                    fraction_evaluate=0.0,  # Disable evaluation for demo purpose
                    min_fit_clients=1,
                    min_available_clients=1,
                    fit_metrics_aggregation_fn=weighted_metrics_average
                )

                client_proxy = CustomClientProxy(cid=client_id)
                results: List[Tuple[ClientProxy, FitRes]] = [
                    (
                        client_proxy,
                        FitRes(
                            status=Status(code=Code.OK, message="Success"),
                            parameters=agg_parameters,
                            num_examples=num_examples,
                            metrics=metrics,  # Use metrics dictionary directly
                            loss=loss  # Add loss if available
                        ),
                    )
                ]
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []

                logger.info(f"---------- call fedavg.aggregate_fit --------------------->")
                parameters_aggregated, metrics_aggregated = fedavg.aggregate_fit(1, results, failures)

                logger.info(f"check parameters_aggregated --------------------->")
                if parameters_aggregated is not None:
                    logger.info(".......................saving parameters_aggregated.......................")
                    save_parameters_aggregated_to_db(workflow_trace_id, domain, parameters_aggregated)
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays = parameters_to_ndarrays(parameters_aggregated)
                    # print("saved parameters_aggregated to db DB Model weights:", aggregated_ndarrays)
                    # Format and print metrics
                    readable_metrics = format_metrics(metrics_aggregated)
                    logger.info("Aggregated Metrics:", readable_metrics)

            else:
                logger.error(f"No training record found for workflow_trace_id {workflow_trace_id}")
        except Exception as e:
            logger.error(f"Error getting model weights with compression: {e}")
            raise


def save_parameters_aggregated_to_db(workflow_trace_id, domain, parameters_aggregated):
    """Save the aggregated parameters to the database."""
    parameters_compressed = compress_weights(parameters_aggregated)
    logger.info(f"save parameters weights to db DB Model weights: {parameters_compressed}")

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

def format_metrics(metrics):
    """Format metrics into a readable string."""
    return "\n".join([f"{key}: {value}" for key, value in metrics.items()])

def weighted_metrics_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}
