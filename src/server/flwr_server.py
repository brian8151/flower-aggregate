from typing import List, Tuple

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics

from src.util import log
logger = log.init_logger()

# Define metric aggregation function
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


# Define strategy
strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Define config
config = ServerConfig(num_rounds=3)

# Flower ServerApp
app = ServerApp(config=config,strategy=strategy,)

#
# # Legacy mode
# if __name__ == "__main__":
#     from flwr.server import start_server
#
#     start_server(
#         server_address="0.0.0.0:8080",
#         config=config,
#         strategy=strategy,
#     )