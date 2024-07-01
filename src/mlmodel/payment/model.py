config = {
    "class_name": "Sequential",
    "config": {
        "name": "sequential",
        "layers": [
            {
                "module": "keras.layers",
                "class_name": "InputLayer",
                "config": {
                    "batch_input_shape": [None, 27],
                    "dtype": "float32",
                    "sparse": False,
                    "ragged": False,
                    "name": "input_1"
                },
                "registered_name": None
            },
            {
                "module": "keras.layers",
                "class_name": "Dense",
                "config": {
                    "name": "dense",
                    "trainable": True,
                    "dtype": "float32",
                    "units": 32,
                    "activation": "tanh",
                    "use_bias": True,
                    "kernel_initializer": {
                        "module": "keras.initializers",
                        "class_name": "GlorotUniform",
                        "config": {"seed": None},
                        "registered_name": None
                    },
                    "bias_initializer": {
                        "module": "keras.initializers",
                        "class_name": "Zeros",
                        "config": {},
                        "registered_name": None
                    },
                    "kernel_regularizer": None,
                    "bias_regularizer": None,
                    "activity_regularizer": None,
                    "kernel_constraint": None,
                    "bias_constraint": None
                },
                "registered_name": None,
                "build_config": {"input_shape": [None, 27]}
            },
            {
                "module": "keras.layers",
                "class_name": "Dense",
                "config": {
                    "name": "dense_1",
                    "trainable": True,
                    "dtype": "float32",
                    "units": 64,
                    "activation": "tanh",
                    "use_bias": True,
                    "kernel_initializer": {
                        "module": "keras.initializers",
                        "class_name": "GlorotUniform",
                        "config": {"seed": None},
                        "registered_name": None
                    },
                    "bias_initializer": {
                        "module": "keras.initializers",
                        "class_name": "Zeros",
                        "config": {},
                        "registered_name": None
                    },
                    "kernel_regularizer": None,
                    "bias_regularizer": None,
                    "activity_regularizer": None,
                    "kernel_constraint": None,
                    "bias_constraint": None
                },
                "registered_name": None,
                "build_config": {"input_shape": [None, 32]}
            },
            {
                "module": "keras.layers",
                "class_name": "Dense",
                "config": {
                    "name": "dense_2",
                    "trainable": True,
                    "dtype": "float32",
                    "units": 2,
                    "activation": "softmax",
                    "use_bias": True,
                    "kernel_initializer": {
                        "module": "keras.initializers",
                        "class_name": "GlorotUniform",
                        "config": {"seed": None},
                        "registered_name": None
                    },
                    "bias_initializer": {
                        "module": "keras.initializers",
                        "class_name": "Zeros",
                        "config": {},
                        "registered_name": None
                    },
                    "kernel_regularizer": None,
                    "bias_regularizer": None,
                    "activity_regularizer": None,
                    "kernel_constraint": None,
                    "bias_constraint": None
                },
                "registered_name": None,
                "build_config": {"input_shape": [None, 64]}
            }
        ]
    },
    "keras_version": "2.15.0",
    "backend": "tensorflow"
}


def get_payment_config():
    return config
