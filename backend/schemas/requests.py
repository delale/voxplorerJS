"""
Pydantic models for request validation.

Defines schemas for:
- Feature extraction requests
- Dimensionality reduction requests
- Data parsing requests
"""

from pydantic import BaseModel, Field
from typing import Optional, Any, List, Dict, Literal


class HealthCheckResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Status of the API")


class APIInfoResponse(BaseModel):
    """API information response schema."""

    title: str = Field(..., description="API title")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    endpoints: dict[str, str] = Field(..., description="Available API endpoints")


# Feature Extraction Schemas (placeholders for future implementation)
class FeatureExtractionRequest(BaseModel):
    """Base schema for feature extraction requests."""

    method: str = Field(..., description="Feature extraction method (e.g., 'mfcc', 'speaker_embedding')")
    description: str = "Placeholder for feature extraction request"


class FeatureExtractionResponse(BaseModel):
    """Response schema for feature extraction endpoint."""

    success: bool = Field(..., description="Whether the extraction was successful")
    features: List[Dict[str, Any]] = Field(..., description="Extracted features as list of dicts (rows)")
    labels: List[str] = Field(..., description="Feature names/labels")
    metadata: Dict[str, Any] = Field(..., description="Extracted metadata (variables from filename, file info)")
    method: str = Field(..., description="Feature extraction method used (mfcc or speaker_embeddings)")
    message: Optional[str] = Field(default=None, description="Additional message or logging info")


# Dimensionality Reduction Schemas (placeholders for future implementation)
class ReductionRequest(BaseModel):
    """Base schema for dimensionality reduction requests."""

    algorithm: str = Field(..., description="Dimensionality reduction algorithm (e.g., 'pca', 'umap', 'tsne', 'mds')")
    n_components: int = Field(default=2, description="Number of output dimensions")
    description: str = "Placeholder for dimensionality reduction request"


class DimensionalityReductionRequest(BaseModel):
    """Request schema for dimensionality reduction endpoint."""

    data: List[List[float]] = Field(
        ...,
        description="2D array of features (n_samples × n_features)"
    )
    algorithm: Literal["pca", "umap", "tsne", "mds"] = Field(
        ...,
        description="Dimensionality reduction algorithm"
    )
    n_components: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Number of output dimensions (1-3)"
    )
    n_neighbors: int = Field(
        default=15,
        ge=2,
        description="Number of neighbors for UMAP (ignored for other algorithms)"
    )
    metric: str = Field(
        default="euclidean",
        description="Distance metric for UMAP and t-SNE"
    )
    supervised: bool = Field(
        default=False,
        description="Use supervised reduction (UMAP only)"
    )
    y: Optional[List[int]] = Field(
        default=None,
        description="Target labels for supervised reduction (required if supervised=True)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    perplexity: Optional[float] = Field(
        default=None,
        description="Perplexity parameter for t-SNE"
    )


class DimensionalityReductionResponse(BaseModel):
    """Response schema for dimensionality reduction endpoint."""

    success: bool = Field(..., description="Whether the reduction was successful")
    algorithm_used: str = Field(..., description="Algorithm used for reduction")
    n_components: int = Field(..., description="Number of output dimensions")
    reduced_data: Dict[str, List[float]] = Field(
        ...,
        description="Reduced dimensions with keys 'DIM1', 'DIM2', optionally 'DIM3'"
    )
    n_samples: int = Field(..., description="Number of samples processed")
    n_features_input: int = Field(..., description="Number of input features")
    explained_variance: Optional[List[float]] = Field(
        default=None,
        description="Explained variance ratio (only for PCA)"
    )
    message: Optional[str] = Field(
        default=None,
        description="Status message or warning"
    )


# Data Parsing Schemas
class DataParsingRequest(BaseModel):
    """Base schema for data parsing requests."""

    format: str = Field(..., description="Data format (e.g., 'csv', 'tsv', 'xlsx')")
    description: str = "Placeholder for data parsing request"


class DataParsingMetadata(BaseModel):
    """Metadata about parsed data."""

    columns: list[str] = Field(..., description="List of column names")
    dtypes: dict[str, str] = Field(..., description="Column name to data type mapping")
    row_count: int = Field(..., description="Number of rows in the table")
    column_count: int = Field(..., description="Number of columns in the table")
    file_name: str = Field(..., description="Original file name")
    file_size: int = Field(..., description="File size in bytes")


class DataParsingResponse(BaseModel):
    """Response schema for data parsing endpoint."""

    success: bool = Field(..., description="Whether parsing was successful")
    data: list[dict[str, Any]] = Field(..., description="List of rows as dictionaries")
    metadata: DataParsingMetadata = Field(..., description="Metadata about the parsed data")
    message: Optional[str] = Field(None, description="Optional status message")
