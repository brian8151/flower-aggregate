import json
import json

class Message:
    def __init__(self, message_id, client_id, strategy, parameters, metrics, num_examples, properties):
        self.message_id = message_id
        self.client_id = client_id
        self.strategy = strategy
        self.parameters = parameters
        self.metrics = metrics
        self.num_examples = num_examples
        self.properties = properties

    def serialize(self):
        """Convert message details into a JSON string."""
        message = {
            "message_id": self.message_id,
            "client_id": self.client_id,
            "strategy": self.strategy,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "num_examples": self.num_examples,
            "properties": self.properties
        }
        return json.dumps(message)

    def deserialize(self, message_str):
        """Convert a JSON string back into a dictionary."""
        return json.loads(message_str)

    def __str__(self):
        """Provide a readable string representation of the message details."""
        return f"Message(ID: {self.message_id}, Client ID: {self.client_id}, Parameters: {self.parameters}, Metrics: {self.metrics}, Number of Examples: {self.num_examples}, Properties: {self.properties})"