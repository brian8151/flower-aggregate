from http.client import HTTPException

from fastapi import APIRouter
from src.service.aggregator_runner_service import AggregatorRunner
from src.model.aggregate_request import AggregatorRequest
from src.util import log

logger = log.init_logger()
aggregate_router = APIRouter()


@aggregate_router.get("/health")
async def check_health():
    try:
        # You can add any specific health check logic here if needed
        return {"message": "Service is healthy", "success": True}
    except Exception as e:
        logger.error("Health check failed: {0}".format(str(e)))
        return {"message": "Service is not healthy", "success": False}


@aggregate_router.post("/run-aggregate")
async def aggregate(request: AggregatorRequest):
    try:
        aggregator_runner = AggregatorRunner()
        aggregator_runner.aggregate(request.workflow_trace_id, request.domain_type)
        return {"status": "success", "domain": request.domain_type, "workflowTraceId": request.workflow_trace_id}
    except Exception as e:
        logger.error(f"Error aggregate fit: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
