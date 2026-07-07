"""
Dimensionality reduction endpoints.

Provides endpoints for:
- PCA reduction
- UMAP reduction
- t-SNE reduction
- MDS reduction
"""

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, status

# Import reduction functions
from lib.dimensionality_reduction import pca_reduction, umap_reduction, tsne_reduction, mds_reduction
from backend.schemas.requests import DimensionalityReductionRequest, DimensionalityReductionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reduction", tags=["reduction"])


def _validate_input_data(data: list[list[float]]) -> tuple[np.ndarray, int, int]:
    """Validate and convert input data to numpy array.
    
    Args:
        data: 2D array as list of lists
        
    Returns:
        Tuple of (numpy array, n_samples, n_features)
        
    Raises:
        ValueError: If data is invalid
    """
    if not data:
        raise ValueError("Input data cannot be empty")
    
    try:
        X = np.array(data, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Data must be convertible to numeric array: {e}")
    
    if X.ndim != 2:
        raise ValueError(f"Data must be 2D (n_samples × n_features), got shape {X.shape}")
    
    n_samples, n_features = X.shape
    
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples, got {n_samples}")
    
    if n_features < 1:
        raise ValueError(f"Need at least 1 feature, got {n_features}")
    
    return X, n_samples, n_features


def _validate_supervised_params(
    algorithm: str,
    supervised: bool,
    y: Optional[list[int]],
    n_samples: int,
) -> Optional[np.ndarray]:
    """Validate supervised learning parameters.
    
    Args:
        algorithm: Reduction algorithm name
        supervised: Whether using supervised mode
        y: Target labels
        n_samples: Number of samples
        
    Returns:
        numpy array of labels or None
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not supervised:
        return None
    
    if algorithm != "umap":
        raise ValueError(f"Supervised mode is only supported for UMAP, got {algorithm}")
    
    if y is None:
        raise ValueError("Target labels (y) are required for supervised reduction")
    
    if len(y) != n_samples:
        raise ValueError(
            f"Length of y ({len(y)}) must match number of samples ({n_samples})"
        )
    
    return np.array(y, dtype=np.int32)


def _convert_to_dict_format(data: np.ndarray, n_components: int) -> dict[str, list[float]]:
    """Convert reduced data to dictionary format with dimension keys.
    
    Args:
        data: Reduced data array (n_samples × n_components)
        n_components: Number of components
        
    Returns:
        Dictionary with keys DIM1, DIM2, optionally DIM3
    """
    result = {}
    dimension_names = ["DIM1", "DIM2", "DIM3"]
    
    for i in range(min(n_components, data.shape[1])):
        result[dimension_names[i]] = data[:, i].tolist()
    
    return result


@router.get("/")
async def reduction_root() -> dict:
    """Root endpoint for dimensionality reduction endpoints."""
    return {
        "message": "Dimensionality reduction endpoints",
        "available_algorithms": ["pca", "umap", "tsne", "mds"],
        "endpoints": {
            "reduce": "POST /api/reduction/reduce"
        }
    }


@router.post("/reduce", response_model=DimensionalityReductionResponse)
async def reduce(
    request: DimensionalityReductionRequest,
) -> DimensionalityReductionResponse:
    """
    Perform dimensionality reduction on input data.
    
    Supports four algorithms:
    - **PCA**: Unsupervised, returns explained variance
    - **UMAP**: Supports supervised mode with target labels
    - **t-SNE**: Unsupervised, good for visualization
    - **MDS**: Unsupervised, preserves pairwise distances
    
    Args:
        request: DimensionalityReductionRequest containing:
            - data: 2D array of features
            - algorithm: 'pca', 'umap', 'tsne', or 'mds'
            - n_components: Output dimensions (1-3, default 2)
            - n_neighbors: UMAP parameter (default 15)
            - metric: Distance metric (default 'euclidean')
            - supervised: Use supervised mode (UMAP only)
            - y: Target labels for supervised UMAP
            - seed: Random seed for reproducibility
            - perplexity: t-SNE perplexity parameter
            
    Returns:
        DimensionalityReductionResponse with reduced dimensions
        
    Raises:
        HTTPException: For validation or processing errors
        
    Examples:
        >>> # PCA example
        >>> request = {
        ...     "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ...     "algorithm": "pca",
        ...     "n_components": 2
        ... }
        >>> response = await reduce(request)
        
        >>> # Supervised UMAP example
        >>> request = {
        ...     "data": [[1, 2], [4, 5], [7, 8]],
        ...     "algorithm": "umap",
        ...     "supervised": True,
        ...     "y": [0, 0, 1]
        ... }
        >>> response = await reduce(request)
    """
    try:
        # Validate input data
        X, n_samples, n_features = _validate_input_data(request.data)
        
        # Validate supervised parameters
        y = _validate_supervised_params(
            request.algorithm,
            request.supervised,
            request.y,
            n_samples,
        )
        
        logger.info(
            f"Reducing dimensions using {request.algorithm} "
            f"(n_samples={n_samples}, n_features={n_features}, "
            f"n_components={request.n_components})"
        )
        
        # Prepare algorithm-specific kwargs
        kwargs = {"n_components": request.n_components}
        
        # Add seed if provided
        if request.seed is not None:
            kwargs["random_state"] = request.seed
        
        # Call appropriate reduction function
        explained_variance = None
        
        if request.algorithm == "pca":
            X_reduced, explained_variance, _ = pca_reduction(X, **kwargs)
            
        elif request.algorithm == "umap":
            umap_kwargs = {
                "n_neighbors": request.n_neighbors,
                "metric": request.metric,
                **kwargs
            }
            if y is not None:
                umap_kwargs["y"] = y
            X_reduced = umap_reduction(X, **umap_kwargs)
            
        elif request.algorithm == "tsne":
            tsne_kwargs = {
                "metric": request.metric,
                **kwargs
            }
            if request.perplexity is not None:
                tsne_kwargs["perplexity"] = request.perplexity
            X_reduced = tsne_reduction(X, **tsne_kwargs)
            
        elif request.algorithm == "mds":
            mds_kwargs = {
                "metric": True,  # Use metric MDS
                **kwargs
            }
            X_reduced = mds_reduction(X, **mds_kwargs)
            
        else:
            raise ValueError(f"Unsupported algorithm: {request.algorithm}")
        
        # Convert to dict format
        reduced_data = _convert_to_dict_format(X_reduced, request.n_components)
        
        # Convert explained variance to list if available
        explained_variance_list = None
        if explained_variance is not None:
            explained_variance_list = explained_variance.tolist()
        
        logger.info(f"Successfully reduced dimensions for {n_samples} samples")
        
        return DimensionalityReductionResponse(
            success=True,
            algorithm_used=request.algorithm,
            n_components=request.n_components,
            reduced_data=reduced_data,
            n_samples=n_samples,
            n_features_input=n_features,
            explained_variance=explained_variance_list,
            message=f"Successfully reduced {n_samples} samples from {n_features} to {request.n_components} dimensions",
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error during dimensionality reduction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during dimensionality reduction: {str(e)}",
        )
