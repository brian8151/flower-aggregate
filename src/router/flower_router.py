from http.client import HTTPException
from fastapi import APIRouter

from src.common.parameter import ndarrays_to_parameters, serialize_parameters
from src.model.client_message_req import ClientMessageRequest
from src.model.client_message_res import ClientMessageResponse

flower_router = APIRouter()
from src.util import log
logger = log.init_logger()
from src.ml.flwr_machine_learning import setup_and_load_data
def fit(parameters, model, x_train, y_train, x_test, y_test):
    model.set_weights(parameters)
    model.fit(x_train, y_train, epochs=1, batch_size=32)
    return model.get_weights(), len(x_train), {}

def client_evaluate(model, parameters, x_test, y_test):
    print(f"---- client_evaluate-----")
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, len(x_test), {"accuracy": accuracy}


@flower_router.post("/send-fedavg")
async def process_fed_avg(message: ClientMessageRequest):
    try:
        client_id = message.client_id
        message_id = message.message_id
        data_path = message.data_path
        # Log or process the received data
        logger.info("Received from client {0}, message id: {1}, data path: {2} ".format(client_id, message_id, data_path))
        # file_path = f'/apps/data/mock_payment_data-0.7.csv'
        # Instantiate FlwrMachineLearning class
        # Setup TensorFlow and load data
        logger.info("rerun model")
        model, x_train, y_train, x_test, y_test = setup_and_load_data(data_path)
        weights = model.get_weights()
        logger.info("Prediction Model weights: {0}".format(weights))
        logger.info("now run fit")
        fit_weights, x_train_length, additional_info = fit(weights, model, x_train, y_train, x_test, y_test)
        logger.info("Fit Model weights: {0}".format(fit_weights))
        loss, num_examples, metrics = client_evaluate(model, weights, x_test, y_test);
        # Print or use the results
        logger.info("Loss: {0}, Number of Test Examples: {1}, Metrics: {2}".format(loss, num_examples, metrics))
        # Serialize the model weights to send
        parameters = ndarrays_to_parameters(weights)
        ser_parameters = serialize_parameters(parameters)
        # Prepare and send the message containing weights and metrics
        res=  ClientMessageResponse(
            message_id=message.message_id,
            client_id=message.client_id,
            strategy="fedavg",
            parameters=ser_parameters,
            metrics=metrics,
            num_examples=num_examples,
            loss=loss,
            properties={"additional_info": additional_info}
        )
        # logger.info("res: {0}".format(res))
        return res
    except Exception as e:
        logger.error(f"Failed to process message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")