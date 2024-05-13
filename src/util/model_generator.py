import tensorflow as tf

def create_model(model_type):
    """Create a model based on the specified type."""
    if model_type == "payment":
        return create_payment_model()
    else:
        raise ValueError("Unsupported model type")

def create_payment_model():
    """Define and compile the payment model."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(27)),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # adjusted from_logits to match the use of softmax in the last layer
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    # Specify the type of model to create
    model_type = 'payment'

    # Create the model based on the specified type
    model = create_model(model_type)

    # Serialize the model to JSON
    model_json = model.to_json()

    # Print the model JSON
    print(model_json)
