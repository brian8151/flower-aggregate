""" Main """

from src.server.flwr_server import app, config, strategy
from flwr.server import start_server
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
from src.util import log
logger = log.init_logger()

def start_flwr_server():
    logger.info("Attempting to start a minimal FLWR server for debugging.")
    try:
        # Minimal strategy setup
        start_server(server_address="0.0.0.0:8080")
        logger.info("If this is logged, the server started and didn't block as expected.")

    except Exception as e:
        logger.error("An error occurred during minimal server start-up: %s", e, exc_info=True)

if __name__ == "__main__":
    start_flwr_server()