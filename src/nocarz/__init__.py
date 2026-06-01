"""Nocarz — location-profitability prediction package.

Modules:
    features        feature engineering shared by training and serving (parity)
    schemas         Pydantic request/response models for the microservice
    routing         deterministic A/B model assignment
    logging_io      thread-safe JSONL request logging
    model_registry  load serialized models + versions
    app             FastAPI application
"""

__version__ = "1.0.0"
