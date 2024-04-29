""" Main """

from src.server.flwr_server import app, config, strategy
from flwr.server import start_server

if __name__ == "__main__":
    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )