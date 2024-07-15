import argparse
from unittest.mock import MagicMock

from src.common.onyx_custom_client_proxy import CustomClientProxy
from src.util import log
logger = log.init_logger()
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Code, FitRes, Status, EvaluateRes, EvaluateIns
from src.ml.flwr_machine_learning import setup_and_load_data
from src.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from src.protocol.simple_message import send_message, receive_message
from flwr.common.typing import List, Tuple, Union
from flwr.common import Metrics
import tensorflow as tf
from flwr.server.client_manager import SimpleClientManager
# In-memory "database"
memory_db = {}
message_queue = []
def fit(parameters, model, x_train, y_train, x_test, y_test):
    model.set_weights(parameters)
    model.fit(x_train, y_train, epochs=1, batch_size=32)
    return model.get_weights(), len(x_train), {}

def client_evaluate(model, parameters, x_test, y_test):
    print(f"---- client_evaluate-----")
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, len(x_test), {"accuracy": accuracy}


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


def client_send_metrics_and_weights(weight, loss, num_examples, metrics, message_queue, client_id):
    print(f"Loss: {loss}")
    print(f"Number of Test Examples: {num_examples}")
    print(f"Metrics: {metrics}")
    # Serialize the model weights to send
    ser_parameters = ndarrays_to_parameters(weight)
    # Prepare and send the message containing weights and metrics
    message = {
        "client_id": client_id,
        "parameters": ser_parameters,
        "metrics": metrics,
        "num_examples": num_examples
    }
    send_message(message_queue, message)
    print("########################### Metrics and weights sent to aggregator with config protocol. ###########################")
def server_receive_metrics_and_weights(message_queue):
    metrics_collected = []
    weights_collected = []

    # Loop to receive messages and process them
    while True:  # Adjust this loop condition based on your actual server control logic
        message = receive_message(message_queue)
        if message is None:
            break  # Exiting the loop when no more messages are available

        # Extract data from the message
        client_id = message['client_id']
        received_parameters = message['parameters']
        received_metrics = message['metrics']
        num_examples = message['num_examples']

        # Deserialize the parameters to model weights if needed
        received_weights = parameters_to_ndarrays(received_parameters)

        # Collect metrics and weights
        metrics_collected.append((num_examples, received_metrics))
        weights_collected.append((num_examples, received_weights))

        print(f"###########################  Received from client {client_id}: Metrics: {received_metrics}, Number of Examples: {num_examples} ###########################")

    # Return collected metrics and optionally weights
    return client_id, num_examples, metrics_collected, weights_collected


def main():
    # Parse arguments to get partition ID and CSV file name
    parser = argparse.ArgumentParser(description="Flower Client Configuration")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Partition of the dataset (0, 1, or 2).",
    )
    parser.add_argument(
        "csv_file_name",
        type=str,
        help="Name of the CSV file to load data from."
    )
    args = parser.parse_args()

    # Construct the file path
    # file_path = f'/apps/data/{args.csv_file_name}'
    file_path1= f'/apps/data/mock_payment_data-0.3.csv'
    print("File path1:", file_path1)
    # Instantiate FlwrMachineLearning class
    # Setup TensorFlow and load data
    model, x_train, y_train, x_test, y_test = setup_and_load_data(args.partition_id, file_path1)
    # Generate client ID
    client_id = f"client_{args.partition_id}"
    print("client_id:", client_id)
    predictions = model.predict(x_test)
    print("Predictions:", predictions)
    # Get model weights
    weights = model.get_weights()
    print("Prediction Model weights:", weights)
    parameters = ndarrays_to_parameters(weights)
    #save to db
    print("save Model weights to db :")
    # Store the parameters in the in-memory database
    memory_db['model_weights'] = parameters
    # Retrieve the parameters from the in-memory database
    print("Retrieve Model weights from db :")
    parameters_from_db = memory_db['model_weights']
    weights_from_db = parameters_to_ndarrays(parameters_from_db)
    print("DB Model weights:", weights_from_db)
    # Set the weights back to your model
    model.set_weights(weights_from_db)
    print("After feedback, we get new data set")
    file_path2= f'/apps/data/mock_payment_data-0.7.csv'
    print("File path2:", file_path2)
    # Instantiate FlwrMachineLearning class
    # Setup TensorFlow and load data
    print("rerun model")
    model1, x_train1, y_train1, x_test1, y_test1 = setup_and_load_data(args.partition_id, file_path2)
    print("now run fit")
    fit_weights, x_train_length, additional_info = fit(weights_from_db, model1, x_train1, y_train1, x_test1, y_test1)
    print("Fit Model weights:", fit_weights)
    loss, num_examples, metrics = client_evaluate(model1, weights_from_db, x_test1, y_test1);
    # Print or use the results
    print(f"Loss: {loss}")
    print(f"Number of Test Examples: {num_examples}")
    print(f"Metrics: {metrics}")
    print("client send to agg with config protocol")
    # Serialize weights to send
    ser_parameters = ndarrays_to_parameters(fit_weights)
    client_send_metrics_and_weights(fit_weights, loss, num_examples, metrics, message_queue, "BANK1")
    # Receive and process message
    client_id, num_examples, metrics_collected, weights_collected = server_receive_metrics_and_weights(message_queue)
    print(f"received fit_weights on another side - clientId: {client_id}, num_examples: {num_examples}")
    # Check if there are collected metrics and weights
    if metrics_collected and weights_collected:
        # Extract just the weights for further processing
        # weights_only = [weights for _, weights in weights_collected]
        weights_only = [weight for _, weights in weights_collected for weight in weights]

        # Check before serialization
        print("Before serialization:")
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
        print("No metrics or weights collected.")


if __name__ == "__main__":
    main()