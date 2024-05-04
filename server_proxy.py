from typing import List, Tuple

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from flwr.server import start_server
from OnyxClientManager import OnyxClientManager
from OnyxCustomStrategy import OnyxCustomStrategy
from OnyxFlowerServer import OnyxFlowerServer


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    try:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        print("Detailed Client Metrics:")
        print("Client | Examples | Accuracy")

        for i, (num_examples, m) in enumerate(metrics):
            print(f"Client {i+1} | {num_examples} | {m['accuracy']:.4f}")

        weighted_avg_accuracy = sum(accuracies) / sum(examples)
        print(f"Weighted Average Accuracy: {weighted_avg_accuracy:.4f}")
        return {"accuracy": weighted_avg_accuracy}
    except Exception as e:
        print(f"Error in weighted_average: {e}")
        return {"accuracy": 0}  # Return a default or fallback value

# Define strategy
strategy = OnyxCustomStrategy(evaluate_metrics_aggregation_fn=weighted_average)
# strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
clientManager = OnyxClientManager()
server = OnyxFlowerServer(clientManager, strategy)
# Define config
config = ServerConfig(num_rounds=1)
# Proxy for start_server


# Legacy mode
if __name__ == "__main__":
    start_server(
        server_address="0.0.0.0:8080",
        server=server,
        config=config,
        strategy=strategy,
        client_manager=clientManager
    )
