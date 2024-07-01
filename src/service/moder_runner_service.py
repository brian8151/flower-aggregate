import numpy as np
from src.mlmodel.model_builder import load_model_from_json_string, compress_weights, build_model, decompress_weights, \
    model_compile
from src.repository.db.db_connection import DBConnection
from src.repository.model.model_data_repositoty import get_model_feature_record, get_model_training_record
from src.repository.model.model_track_repository import get_model_track_record, create_model_track_records, \
    create_local_model_historical_records, update_workflow_model_process, create_workflow_model_process, \
    save_mode_training_result, create_and_update_model_weight_records, update_local_model_track
from src.tool.model_gen import generate_model_json
from src.util import log

logger = log.init_logger()


class ModelRunner:
    """ Class for machine learning service """

    def __init__(self):
        pass

    def get_model_weights(self, model_json: str):
        try:
            if not model_json:
                raise ValueError("Model JSON is empty")
            model = load_model_from_json_string(model_json)
            weights = model.get_weights()
            logger.info("Model weights retrieved successfully.")
            return weights
        except Exception as e:
            logger.error(f"Error getting model weights: {e}")
            raise

    def get_domain_model_weights(self, domain):
        try:
            model = build_model(domain)
            weights = model.get_weights()
            # logger.info("get model weight: {0}".format(weights))
            return weights
        except Exception as e:
            logger.error(f"Error getting model weights: {e}")
            raise

    def get_model_weights_with_serialize(self, model_json: str):
        try:
            model_weights = self.get_model_weights(model_json)
            # Convert numpy arrays to lists for JSON serialization
            weights_serializable = [w.tolist() for w in model_weights]
            return weights_serializable
            return weights
        except Exception as e:
            logger.error(f"Error getting model weights with serialize: {e}")
            raise

    def get_model_weights_req(self, name, domain, model_version, model_json: str):
        logger.info(f"get_model_weights for domain '{domain}', name: '{name}', model_version: '{model_version}'")
        try:
            model_track_record = get_model_track_record(domain)
            if not model_track_record:
                logger.info(f"No global model track found for domain '{domain}'. Creating a new entry.")
                model_weights = self.get_model_weights(model_json)
                # Compress and encode weights
                logger.info("Compress and encode weights '{0}'.".format(domain))
                weights_compressed = compress_weights(model_weights)
                return weights_compressed
            else:
                local_model_weights = model_track_record[2]
                return local_model_weights
        except Exception as e:
            logger.error(f"Error getting model weights with compression: {e}")
            raise

    def initial_weights(self, name, domain, model_version):
        """
        Initialize weights for the given domain. If no global model track exists for the domain,
        it creates an entry with the same model and weight.

        Parameters:
            name (str): The name of model.
            domain (str): The domain for which to initialize weights.
            model_version (str): The version of the model.
        Returns:
            str: Compressed and encoded model weights.

        Raises:
            Exception: If there is an error in getting or processing model weights.
        """
        try:
            model_track_record = get_model_track_record(domain)
            if not model_track_record:
                logger.info(f"No global model track found for domain '{domain}'. Creating a new entry.")
                local_weights_version = 1
                model_json = generate_model_json(domain)
                model_weights = self.get_model_weights(model_json)
                # Compress and encode weights
                logger.info("Compress and encode weights '{0}'.".format(domain))
                weights_compressed = compress_weights(model_weights)
                logger.info("saving model track records for domain '{0}'.".format(domain))
                create_model_track_records(name, model_json, model_version, domain, weights_compressed,
                                           local_weights_version)
                logger.info("model track records for domain '{0}' saved.".format(domain))
                create_local_model_historical_records("0000000000000000000000000000", name, weights_compressed)
                logger.info("local model historical records for domain '{0}' saved.".format(domain))
                return weights_compressed
            else:
                local_model_weights = model_track_record[2]
                return local_model_weights
        except Exception as e:
            logger.error(f"Error getting model weights with compression: {e}")
            raise

    def get_model_weights_with_compression(self, model_json: str):
        try:
            model_weights = self.get_model_weights(model_json)
            # Compress and encode weights
            weights_compressed = compress_weights(model_weights)
            return weights_compressed
        except Exception as e:
            logger.error(f"Error getting model weights with compression: {e}")
            raise

    def run_model_predict(self, workflow_trace_id, domain_type, batch_id):
        """
          Retrieves model feature records, makes predictions using the given model, and prepares the result.

          Args:
              domain_type (str): The domain type to filter the records.
              batch_id (str): The batch_id to filter the records.
              workflow_trace_id (object): workflow trace id

          Returns:
              list: A list of dictionaries containing data and prediction results.
          """
        logger.info("predict - Build model for domain {0}, workflow_trace_id: {1}".format(domain_type, workflow_trace_id))
        try:
            model = build_model(domain_type)
            logger.info("Model summary: {0}".format(model.summary()))
            model_track_record = get_model_track_record(domain_type)
            local_model_weights = model_track_record[2]
            global_model_weights = model_track_record[4]
            if global_model_weights is None:
                logger.info("global_model_weights is empty, use default local model weight")
                weights_encoded = local_model_weights
            else:
                logger.info("found global_model_weights")
                weights_encoded = global_model_weights
            logger.info("Decompress and decode weights '{0}'.".format(domain_type))
            weights = decompress_weights(weights_encoded)
            model.set_weights(weights)
            logger.info("get model feature records for batch: {0}".format(batch_id))
            data = get_model_feature_record(domain_type, batch_id)
            # Check if data is retrieved successfully
            if not data:
                logger.info("No data found for domain: {0}, batch_id: {1}".format(domain_type, batch_id))
                return []
            logger.info("found model feature records: {0} for batch: {1}".format(len(data), batch_id))
            # Log the columns retrieved and their count
            logger.info(f"Columns retrieved: {len(data[0]) if data else 0}")
            # if data:
            #     for i, col in enumerate(data[0]):
            #         logger.info(f"Column {i}: {col}")
            # Prepare features for prediction
            features = [list(row[1:]) for row in data]
            item_ids = [row[0] for row in data]  # Extract the first column (payment_id)
            # Print the shape of the features array
            logger.info(f"Shape of features array before conversion: {np.array(features).shape}")
            # Convert features to a NumPy array and ensure the correct data type
            features_array = np.array(features, dtype=np.float32)
            # Make predictions
            logger.info("Make predictions")
            y_hat = model.predict(features_array)
            n = len(features)
            logger.info("Sample size: {0}".format(n))

            # Prepare the result
            data_req = [{"itemId": item_ids[i], "result": None} for i in range(n)]
            for i in range(n):
                data_req[i]["result"] = float(100.0 * y_hat[i][0])  # acceptable percentage

            sql_update_query = "UPDATE model_predict_data SET result = %s WHERE item_id = %s"
            values = [(item["result"], item["itemId"]) for item in data_req]

            # Execute the batch insert
            logger.info("Executing batch update for Predict  Data")
            total_records = DBConnection.execute_batch_insert(sql_update_query, values)
            logger.info("Total records updated: {0}".format(total_records))
            update_workflow_model_process(workflow_trace_id, 'OFL-C2', 'Complete')
            return 'success', workflow_trace_id, n
        except Exception as e:
            logger.error(f"Error run model predict workflow-trace_id: {workflow_trace_id}: {e}")
            update_workflow_model_process(workflow_trace_id, 'OFL-C2', 'Fail')
            return 'fail', workflow_trace_id, 0

    def run_model_training(self, workflow_trace_id, domain_type, batch_id):
        """
        Retrieves model feature records, model training using the given model, and prepares the result.

        Args:
            domain_type (str): The domain type to filter the records.
            batch_id (str): The batch_id to filter the records.
            workflow_trace_id (object): workflow trace id

        Returns:
            tuple: A tuple containing workflow_trace_id, loss, number of examples, and metrics.
        """
        logger.info("training - Build model for domain {0}, workflow_trace_id: {1}, batchId: {2}".format(domain_type, workflow_trace_id, batch_id))
        try:
            model = build_model(domain_type)
            logger.info("Model summary: {0}".format(model.summary()))
            model_track_record = get_model_track_record(domain_type)
            local_model_weights = model_track_record[2]
            global_model_weights = model_track_record[4]
            if global_model_weights is None:
                logger.info("global_model_weights is empty, use default local model weight")
                weights_encoded = local_model_weights
            else:
                logger.info("found global_model_weights")
                weights_encoded = global_model_weights
            logger.info("Decompress and decode weights '{0}'.".format(domain_type))
            weights = decompress_weights(weights_encoded)
            model.set_weights(weights)
            logger.info("get model feature records for batch: {0}".format(batch_id))
            data = get_model_training_record(domain_type, batch_id)
            # Check if data is retrieved successfully
            if not data:
                logger.info("No data found for domain: {0}, batch_id: {1}".format(domain_type, batch_id))
                return workflow_trace_id, None, 0, None
            logger.info("found model training records: {0} for batch: {1}".format(len(data), batch_id))
            # Log the columns retrieved and their count
            logger.info(f"Columns retrieved: {len(data[0]) if data else 0}")

            # Prepare features and labels for training and testing
            features = [list(row[1:-3]) for row in
                        data]  # Exclude id_field and the last 3 columns (result, is_correct, score)
            result_list = [float(row[-3]) for row in data]  # Extract result column and cast to float
            is_correct_req = [row[-2] for row in data]  # Extract is_correct column

            # Convert features to NumPy arrays
            features_array = np.array(features, dtype=np.float32)

            y = []
            for i in range(len(result_list)):
                if is_correct_req[i] == "Y":
                    if result_list[i] > 73.0:
                        y.append(0)
                    else:
                        y.append(1)
                else:
                    if result_list[i] > 73.0:
                        y.append(1)
                    else:
                        y.append(0)
            labels_array = np.array(y, dtype=np.int32)

            # Split the data into training and testing sets
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(features_array, labels_array, test_size=0.2,
                                                                random_state=42)

            # Compile the model
            model_compile(model)

            # Train the model
            logger.info("Training the model")
            model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)

            # Evaluate the model
            logger.info("Evaluating the model")
            loss, accuracy = model.evaluate(x_test, y_test)
            num_examples = len(x_test)
            metrics = {"accuracy": accuracy}

            logger.info(f"Loss: {loss}")
            logger.info(f"Number of Test Examples: {num_examples}")
            logger.info(f"Metrics: {metrics}")
            logger.info(f"create workflow model process - OFL-C4: {workflow_trace_id}")
            create_workflow_model_process(workflow_trace_id, 'OFL-C4', 'Complete')
            # Save the model training result
            logger.info(f"save mode training result: {workflow_trace_id}, num_examples: {num_examples}")
            save_mode_training_result(workflow_trace_id, loss, num_examples, metrics)
            weights_compressed = compress_weights(weights)
            logger.info(f"save model weight records: {workflow_trace_id}, num_examples: {num_examples}")
            create_and_update_model_weight_records(workflow_trace_id, domain_type, weights_compressed)
            return 'success', workflow_trace_id, loss, num_examples, metrics

        except Exception as e:
            logger.error(f"Error run model predict workflow-trace_id: {workflow_trace_id}: {e}")
            create_workflow_model_process(workflow_trace_id, 'OFL-C4', 'Fail')
            return 'fail', workflow_trace_id, None, 0, None

