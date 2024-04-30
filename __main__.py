""" Main """

from src.server.flwr_server import app, config, strategy
from flwr.server import start_server
from src.util import log
logger = log.init_logger()

if __name__ == "__main__":
    logger.info("Starting the FLWR server with the specified configuration and strategy.")
    try:
        start_server(
            server_address="0.0.0.0:8080",
            config=config,
            strategy=strategy,
        )
        logger.info("Server started successfully.")
    except Exception as e:
        logger.error(f"An error occurred when starting the server: {e}")