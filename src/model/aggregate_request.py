from pydantic import BaseModel, Field
from typing import List, Optional


class AggregatorRequest(BaseModel):
    domain_type: str = Field(..., alias="domainType", description="data seed domain type")
    workflow_trace_id: str = Field(..., alias="workflowTraceId", description="workflow trace id")
    model_id: str = Field(..., alias="modelId", description="model id")
    group_hash: str = Field(..., alias="groupHash", description="group Hash")

