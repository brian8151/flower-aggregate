import tensorflow as tf
from src.util import log

logger = log.init_logger()


class InvalidDomainError(Exception):
    pass


def compile_build_model(domain=None):
    logger.info("Build for domain {} model".format(domain))
    model = None
    if domain == "payment":
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(27,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2, activation="softmax")
        ])
    elif domain == "domain2":
        logger.info("Build for domain2 model")

    if model is None:
        raise InvalidDomainError(f"Domain '{domain}' is not recognized. Please provide a valid domain.")

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def generate_model_json(domain=None):
    try:
        model = compile_build_model(domain)
        # Convert the model to JSON
        model_json = model.to_json()
        return model_json
    except InvalidDomainError as e:
        logger.error(f"Error generating model JSON: {e}")
        raise
