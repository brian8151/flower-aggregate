from pydantic import BaseModel, Field
from typing import List, Optional


class DataItem(BaseModel):
    features: List[float] = Field(..., description="Prediction data features")


class PredictRequest(BaseModel):
    domain_type: str = Field(..., alias="domainType", description="data seed domain type")
    workflow_trace_id: str = Field(..., alias="workflowTraceId", description="workflow trace id")
    batch_id: str = Field(..., alias="batchId", description="batch id")


class TraningRequest(BaseModel):
    domain_type: str = Field(..., alias="domainType", description="data seed domain type")
    workflow_trace_id: str = Field(..., alias="workflowTraceId", description="workflow trace id")
    batch_id: str = Field(..., alias="batchId", description="batch id")


class ModelInitRequest(BaseModel):
    domain_type: str = Field(..., alias="domainType", description="data seed domain type")
