from pydantic import BaseModel
from typing import Optional, Any, Dict, List


class ECSLog(BaseModel):
    source: Optional[Dict[str, Any]] = None
    event: Optional[Dict[str, Any]] = None
    host: Optional[Dict[str, Any]] = None
    http: Optional[Dict[str, Any]] = None
    url: Optional[Dict[str, Any]] = None
    user_agent: Optional[Dict[str, Any]] = None
    log: Optional[Dict[str, Any]] = None
    ecs: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    type: Optional[str] = None
    timestamp: Optional[str] = None

    class Config:
        extra = "allow"   # 🔥 КРИТИЧНО


class OnlinePredictRequest(BaseModel):
    log: ECSLog
    model_version: Optional[str] = None
    return_proba: bool = False
    write_to_es: bool = False
    source_doc_id: Optional[str] = None
    source_index: Optional[str] = None
    results_index_prefix: Optional[str] = None


class OnlinePredictResponse(BaseModel):
    cluster: Optional[int] = None
    proba: Optional[float] = None
    model_version: Optional[str] = None
    detail: Optional[str] = None
    elapsed: Optional[float] = None


class OfflineTrainRequest(BaseModel):
    source_paths: List[str]
    model_version: Optional[str] = None
    hdbscan_params: Optional[Dict[str, Any]] = None
    lr_params: Optional[Dict[str, Any]] = None
    persist_path: str = "app/src/models"


class OfflineTrainResponse(BaseModel):
    model_version: str
    status: str
    detail: Optional[str] = None


class BatchClassifyRequest(BaseModel):
    raw_index_pattern: Optional[str] = None
    results_index_prefix: Optional[str] = None
    size: int = 100
    query: Optional[Dict[str, Any]] = None


class BatchClassifyResponse(BaseModel):
    processed: int
    indexed: int
    errors: Optional[List[str]] = None
