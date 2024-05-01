""" Main """

from src.server.flwr_server import app, config, strategy
from flwr.server import start_server
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
from src.util import log
logger = log.init_logger()

def start_flwr_server():
    logger.info("Starting the FLWR server with the specified configuration and strategy.")
    try:
        strategy = FedAvg(
            fraction_fit=1.0,
            min_available_clients=1,
            min_fit_clients=1,
            on_evaluate_config_fn=lambda rnd: {"rnd": rnd},
            evaluate_metrics_aggregation_fn=lambda metrics: {
                "accuracy": sum(m["accuracy"] * m["num_examples"] for m in metrics) / sum(
                    m["num_examples"] for m in metrics)}
        )

        config = ServerConfig(num_rounds=3)
        server_address = "0.0.0.0:8080"
        logger.info(f"Server starting on {server_address}")
        start_server(
            server_address=server_address,
            config=config,
            strategy=strategy,
        )
        import time
        while True:
            logger.info("Server is running...")
            time.sleep(60)  # Keep the server running by sleeping in a loop
    except Exception as e:
        logger.error(f"An error occurred when starting the server: {e}", exc_info=True)


if __name__ == "__main__":
    start_flwr_server()