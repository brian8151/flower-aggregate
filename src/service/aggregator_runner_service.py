import numpy as np
from src.util import log
from src.repository.model.model_data_repositoty import get_model_client_training_record, save_model_aggregate_result, get_model_info
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
                logger.info(f"Retrieved {domain} training record: {training_record}")
                logger.info(f"Training record keys: {list(training_record.keys())}")
                client_id = training_record['client_id']
                model_id = training_record['model_id']
                weights_encoded = training_record['parameters']
                num_examples = training_record['num_examples']
                accuracy = training_record.get('accuracy')

                if not all([client_id, weights_encoded, num_examples, accuracy is not None]):
                    logger.error("Missing required keys in training_record")
                    raise KeyError("Missing required keys in training_record")

                # Decompress weights
                decompressed_weights = decompress_weights(weights_encoded)
                ser_parameters = ndarrays_to_parameters(decompressed_weights)
                weights_as_ndarrays = parameters_to_ndarrays(ser_parameters)

                metrics = {"accuracy": accuracy}
                metrics_collected = []
                weights_collected = []
                metrics_collected.append((num_examples, metrics))
                weights_collected.append((num_examples, weights_as_ndarrays))

                logger.info(f"metrics_collected: {metrics_collected}")
                weights_only = [weight for _, weights in weights_collected for weight in weights]

                logger.info(f"loop weights_only")
                for idx, weight in enumerate(weights_only):
                    logger.info(f"Weight {idx} shape: {weight.shape}, dtype: {weight.dtype}")

                agg_parameters = ndarrays_to_parameters(weights_only)
                # Aggregate metrics
                aggregated_metrics = weighted_average(metrics_collected)
                logger.info(f"Aggregated Metrics: {aggregated_metrics}")

                fedavg = FedAvg(
                    fraction_fit=0.2,
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
                        ),
                    )
                ]
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
                logger.info(f"---------- call fedavg.aggregate_fit --------------------->")
                parameters_aggregated, metrics_aggregated = fedavg.aggregate_fit(1, results, failures)
                if parameters_aggregated is not None:
                    group_hash= "abcde123"
                    logger.info(f"saving  parameters_aggregated, model id:{model_id}")
                    save_parameters_aggregated_to_db(workflow_trace_id, client_id, model_id, group_hash, parameters_aggregated, metrics_aggregated, num_examples)
                    readable_metrics = format_metrics(metrics_aggregated)
                    logger.info(f"Aggregated Metrics: {readable_metrics}")
                else:
                    logger.error(f"No fed strategy parameters from fedavg for workflow_trace_id {workflow_trace_id}")
            else:
                logger.error(f"No training record found for workflow_trace_id {workflow_trace_id}")
        except Exception as e:
            logger.error(f"Error getting model weights with compression: {e}")
            raise


def save_parameters_aggregated_to_db(workflow_trace_id, client_id, model_id,group_hash, parameters_aggregated, metrics_aggregated, num_examples):
    """Save the aggregated parameters to the database."""
    parameters_compressed = compress_weights(parameters_aggregated)
    logger.info(f"{workflow_trace_id}  save parameters weights to db DB Model weights")
    loss = metrics_aggregated.get("loss", 0.0)
    accuracy = metrics_aggregated.get("accuracy", 0.0)
    metrics = {"accuracy": accuracy, "loss": loss}
    save_model_aggregate_result(workflow_trace_id, client_id, model_id, group_hash, loss, num_examples, metrics, parameters_compressed)

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
    # Handle missing loss keys
    losses = [num_examples * m.get("loss", 0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    total_examples = sum(examples)
    avg_accuracy = sum(accuracies) / total_examples
    avg_loss = sum(losses) / total_examples if sum(losses) > 0 else None

    result = {"accuracy": avg_accuracy}
    if avg_loss is not None:
        result["loss"] = avg_loss
    return result