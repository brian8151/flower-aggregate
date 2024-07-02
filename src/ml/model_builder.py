import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
import io
import contextlib
import pickle
import gzip
import base64

from src.util import log

logger = log.init_logger()


@register_keras_serializable()
class Sequential(tf.keras.Sequential):
    pass


def capture_model_summary(model):
    # This function captures the model summary
    try:
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        return "\n".join(summary_list)
    except Exception as e:
        logger.error(f"Error capturing model summary: {e}")
        return "Error capturing model summary."



def load_model_from_json_string(model_json: str):
    try:
        # custom_objects = {'Sequential': Sequential}
        # model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
        model = tf.keras.models.model_from_json(model_json)
        # model = tf.keras.models.model_from_json(model_json)
        model_summary = capture_model_summary(model)
        logger.info("Model architecture loaded successfully.\nModel Summary:\n{0}".format(model_summary))
        return model
    except Exception as e:
        logger.error(f"Error loading model from JSON: {e}")
        raise


def compress_weights(weights):
    logger.info("Compressing weights...")
    weights_serialized = pickle.dumps(weights)
    weights_compressed = gzip.compress(weights_serialized)
    weights_encoded = base64.b64encode(weights_compressed).decode('utf-8')
    logger.info(f"Weights compressed and encoded: {weights_encoded[:100]}...")  # Log first 100 characters
    return weights_encoded


def decompress_weights(weights_encoded):
    """
    Decompress the model weights using base64, gzip, and pickle.

    Args:
        weights_encoded (str): The compressed and encoded weights.

    Returns:
        The decompressed model weights.
    """
    logger.info(f"Decompressing weights. Input type: {type(weights_encoded)}, size: {len(weights_encoded)}")
    try:
        # Step 1: Decode from base64
        weights_compressed = base64.b64decode(weights_encoded)
        logger.debug(f"Decoded weights. Type: {type(weights_compressed)}, size: {len(weights_compressed)}")

        # Step 2: Check if data is gzip compressed
        if is_gzip_compressed(weights_compressed):
            logger.info("Data is gzip compressed. Proceeding with gzip decompression.")
            try:
                weights_serialized = gzip.decompress(weights_compressed)
                logger.info(f"Decompressed weights. Type: {type(weights_serialized)}, size: {len(weights_serialized)}")
            except OSError as e:
                logger.error(f"Error during gzip decompression: {e}")
                raise
        else:
            logger.info("Data is not gzip compressed. Skipping gzip decompression.")
            weights_serialized = weights_compressed

        # Step 3: Deserialize using pickle
        weights = pickle.loads(weights_serialized)
        logger.debug("Weights decompressed and deserialized successfully.")
        return weights
    except Exception as e:
        logger.error(f"Error during decompression: {e}")
        raise

def is_gzip_compressed(data):
    """
    Check if the data is in gzip format.

    Args:
        data (bytes): The data to check.

    Returns:
        bool: True if the data is gzip compressed, False otherwise.
    """
    return data[:2] == b'\x1f\x8b'