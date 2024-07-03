from src.repository.db.db_connection import DBConnection
from src.util import log
from datetime import datetime

logger = log.init_logger()


def get_model_info(domain):
    """
    Returns an empty tuple if no record is found.
    """
    sql = (
        """SELECT id, model_name, model_definition FROM model_definition WHERE domain='{}'"""
        .format(domain))
    result = DBConnection.execute_query(sql)
    if result:
        return result[0][0], result[0][1], result[0][2]
    return ()

def get_model_client_training_record(workflow_trace_id, domain):
    """
    Retrieve the training record for a given domain and workflow trace ID.

    This function fetches the client training record from the database
    by executing an SQL query that joins the `model_client_training_result`
    and `metrics` tables. The query returns a single row that matches the
    provided `workflow_trace_id`.

    Parameters:
    domain (str): The domain for which the record is being retrieved.
    workflow_trace_id (str): The unique identifier for the workflow trace.

    Returns:
    dict: A dictionary containing the client training record with keys:
          - 'client_id': ID of the client
          - 'model_id': ID of the model
          - 'parameters': Model parameters
          - 'loss': Training loss
          - 'num_examples': Number of examples used in training
          - 'accuracy': Model accuracy
    None: If no record is found for the given `workflow_trace_id`.
    """
    sql = """
        SELECT mr.client_id, mr.model_id, mr.parameters, mr.loss, mr.num_examples, m.accuracy  
        FROM model_client_training_result mr, metrics m
        WHERE workflow_trace_id='{0}' AND mr.metrics_id = m.id
        LIMIT 1
    """.format(workflow_trace_id)

    # Execute the query and fetch the results
    rows = DBConnection.execute_query(sql)
    logger.info("get_model_client_training_record rows: {0}".format(len(rows)))

    if rows:
        record = rows[0]
        return {
            "client_id": record[0],
            "model_id": record[1],
            "parameters": record[2],
            "loss": record[3],
            "num_examples": record[4],
            "accuracy": record[5]
        }
    else:
        logger.error("No record found for domain: {0}, workflow_trace_id: {1}".format(domain, workflow_trace_id))
        return None


def save_model_aggregate_result(workflow_trace_id, client_id, model_id, group_hash, loss, num_examples, metrics,
                                parameters):
    try:
        connection = DBConnection.get_connection()
        if connection.is_connected():
            cursor = connection.cursor()

            # Insert metrics
            loss = round(loss, 10)  # Round the loss value to 10 decimal places
            accuracy = round(metrics['accuracy'], 10)  # Round the accuracy value to 10 decimal places
            insert_metrics_query = "INSERT INTO metrics (accuracy) VALUES (%s)"
            cursor.execute(insert_metrics_query, (accuracy,))
            metrics_id = cursor.lastrowid
            logger.info("save metrics - accuracy")
            # Insert model aggregate weights result
            insert_aggregate_weights_query = """INSERT INTO model_aggregate_weights (workflow_trace_id, model_id, parameters) VALUES (%s, %s, %s)"""
            cursor.execute(insert_aggregate_weights_query, (workflow_trace_id, model_id, parameters))
            model_weights_id = cursor.lastrowid
            logger.info("save aggregate weights")
            # Insert run model aggregation
            insert_run_model_aggregation_query = """
                INSERT INTO run_model_aggregation 
                (workflow_trace_id, client_id, model_id, group_hash, model_weights_id, loss, num_examples, metrics_id, status) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_run_model_aggregation_query, (
            workflow_trace_id, client_id, model_id, group_hash, model_weights_id, loss, num_examples, metrics_id,
            'Complete'))
            logger.info("save run model aggregation")
            # Find model_collaboration_run by group_hash
            find_collaboration_run_query = "SELECT id, current_round, rounds FROM model_collaboration_run WHERE group_hash = %s"
            cursor.execute(find_collaboration_run_query, (group_hash,))
            collaboration_run = cursor.fetchone()

            if collaboration_run:
                collaboration_run_id, current_round, rounds = collaboration_run
                new_round = current_round + 1

                # Update current_round and set started_at if initial round
                update_collaboration_run_query = """
                             UPDATE model_collaboration_run 
                             SET current_round = %s, 
                                 started_at = IF(current_round = 0, %s, started_at), 
                                 status = IF(%s = rounds, 'Complete', status), 
                                 completed_at = IF(%s = rounds, %s, completed_at)
                             WHERE id = %s
                         """
                cursor.execute(update_collaboration_run_query, (
                    new_round, datetime.now(), new_round, new_round, datetime.now(), collaboration_run_id))
                logger.info("update model collaboration run")
            connection.commit()
            logger.info("Model aggregate result saved successfully")
    except Error as e:
        logger.error(f"Error saving model aggregate result: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            logger.info("MySQL cursor and connection are closed")