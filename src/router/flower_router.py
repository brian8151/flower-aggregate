from fastapi import APIRouter

from src.model.message_req import MessageRequest
from src.service.flower.flower_fedavg_service import FlowerFedAvgService
flower_router = APIRouter()
from src.util import log
logger = log.init_logger()
@flower_router.post("/fedavg")
async def process_fed_avg(message: MessageRequest):
    # Log or process the received data
    logger.info("Received from client {0}: Metrics: {1}, Number of Examples: {2}".format(message.client_id, message.metrics, message.num_examples))
    flower_fed_avg_svc = FlowerFedAvgService()
    flower_fed_avg_svc.run_aggregate_fit(message)
    return {"message": "Workflow route"}
