"""
Feature extraction endpoints for audio and tabular data.

Provides endpoints for:
- Extracting MFCCs and speaker embeddings from audio files
- Processing feature extraction requests
"""

import io
import logging
from typing import Optional, Any, Dict, List

from fastapi import APIRouter, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse

from lib.feature_extraction import FeatureExtractor
from backend.schemas.requests import FeatureExtractionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/features", tags=["features"])


@router.get("/")
async def features_root():
    """Root endpoint for feature extraction endpoints."""
    return {"message": "Feature extraction endpoints - extract audio features using /extract"}


@router.post("/extract", response_model=FeatureExtractionResponse)
async def extract_features(
    file: UploadFile,
    method: str = Form(..., description="Feature extraction method: 'mfcc' or 'speaker_embeddings'"),
    n_mfcc: int = Form(default=13, description="Number of MFCCs to extract (for MFCC method)"),
    n_mels: int = Form(default=40, description="Number of mel bands (for MFCC method)"),
    win_length: float = Form(default=25.0, description="Window length in ms (for MFCC method)"),
    overlap: float = Form(default=10.0, description="Overlap in ms (for MFCC method)"),
    fmin: float = Form(default=100.0, description="Minimum frequency (for MFCC method)"),
    fmax: float = Form(default=6000.0, description="Maximum frequency (for MFCC method)"),
    preemphasis: float = Form(default=0.95, description="Preemphasis coefficient (for MFCC method)"),
    lifter: float = Form(default=22.0, description="Liftering coefficient (for MFCC method)"),
    summarise: bool = Form(default=False, description="Summarise features by utterance"),
    include_delta: bool = Form(default=False, description="Include delta features"),
    include_delta_delta: bool = Form(default=False, description="Include delta-delta features"),
    model: str = Form(default="speechbrain/spkrec-ecapa-voxceleb", description="Speaker embedding model (for embeddings method)"),
    metadata_separator: str = Form(default="_", description="Character to split filename for metadata extraction"),
    metadata_variables: Optional[str] = Form(default=None, description="Comma-separated metadata variables to extract from filename"),
) -> FeatureExtractionResponse:
    """
    Extract acoustic features from an audio file.

    This endpoint extracts either MFCCs (Mel-Frequency Cepstral Coefficients) or
    speaker embeddings from uploaded audio files. Supports WAV, MP3, and FLAC formats.

    **Parameters:**
    - **file**: Audio file to process (WAV, MP3, or FLAC)
    - **method**: Feature extraction method - either 'mfcc' or 'speaker_embeddings'
    - **n_mfcc**: Number of MFCC coefficients (default: 13)
    - **summarise**: If True, returns mean and std of features across frames (default: False)
    - **include_delta**: Include first-order delta features (default: False)
    - **include_delta_delta**: Include second-order delta-delta features (default: False)
    - **metadata_separator**: Character to split filename to extract metadata (default: '_')
    - **metadata_variables**: Comma-separated variable names to extract from filename

    **Response:**
    Returns JSON with:
    - **features**: List of feature vectors (one per frame, or one per file if summarise=True)
    - **labels**: Feature names (c1, c2, ... for MFCC; d1, d2, ... for delta; X1, X2, ... for embeddings)
    - **metadata**: Extracted metadata from filename and file info
    - **method**: The method used for extraction
    - **message**: Optional message with additional info

    **Example:**
    ```bash
    curl -X POST "http://localhost:5000/api/features/extract" \\
      -F "file=@audio.wav" \\
      -F "method=mfcc" \\
      -F "n_mfcc=13" \\
      -F "summarise=true"
    ```
    """
    try:
        # Validate method
        if method not in ["mfcc", "speaker_embeddings"]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid method '{method}'. Supported methods: 'mfcc', 'speaker_embeddings'"
            )

        # Read file contents
        try:
            file_contents = await file.read()
        except Exception as e:
            logger.error(f"Failed to read uploaded file: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read uploaded file: {str(e)}"
            )

        if not file_contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )

        # Prepare filebytes in base64 format expected by FeatureExtractor
        import base64
        file_b64 = base64.b64encode(file_contents).decode('utf-8')
        
        # Determine media type from file extension
        filename_lower = file.filename.lower() if file.filename else "audio.wav"
        if filename_lower.endswith('.mp3'):
            media_type = "audio/mpeg"
        elif filename_lower.endswith('.flac'):
            media_type = "audio/flac"
        else:
            media_type = "audio/wav"
        
        filebytes = [f"data:{media_type};base64,{file_b64}"]

        # Parse metadata variables
        metavars_list = None
        if metadata_variables and metadata_variables.strip():
            metavars_list = [v.strip() for v in metadata_variables.split(",")]

        # Prepare feature methods dictionary
        feature_methods = {}
        if method == "mfcc":
            feature_methods["mel_features"] = {
                "n_mfccs": n_mfcc,
                "n_mels": n_mels,
                "win_length": win_length,
                "overlap": overlap,
                "fmin": fmin,
                "fmax": fmax,
                "preemphasis": preemphasis,
                "lifter": lifter,
                "delta": include_delta,
                "delta_delta": include_delta_delta,
                "summarise": summarise,
            }
        elif method == "speaker_embeddings":
            feature_methods["speaker_embeddings"] = {
                "model": model,
            }

        # Prepare metadata variables
        metavars = {
            "variables": metavars_list,
            "split_char": metadata_separator,
        }

        # Create FeatureExtractor instance
        try:
            extractor = FeatureExtractor(
                filenames=[file.filename or "audio.wav"],
                filebytes=filebytes,
                feature_methods=feature_methods,
                metavars=metavars,
            )
        except Exception as e:
            logger.error(f"Failed to initialize FeatureExtractor: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process audio file: {str(e)}"
            )

        # Extract features
        try:
            features_dict, metadata_dict = extractor.process_files()
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Feature extraction failed: {str(e)}"
            )

        # Convert features dictionary to list of dicts (rows format)
        # features_dict has keys as feature names and values as 1D arrays
        # We need to transpose this to rows format
        import numpy as np
        
        feature_labels = list(features_dict.keys())
        
        # Build list of rows
        if feature_labels:
            # Get first feature to determine number of rows
            first_feature = features_dict[feature_labels[0]]
            n_rows = len(first_feature) if hasattr(first_feature, '__len__') else 1
            
            # Build rows
            features_list = []
            for i in range(n_rows):
                row = {}
                for label in feature_labels:
                    feature_data = features_dict[label]
                    row[label] = float(feature_data[i]) if hasattr(feature_data, '__getitem__') else float(feature_data)
                features_list.append(row)
        else:
            features_list = []

        # Clean metadata (convert numpy types to Python types for JSON serialization)
        def clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
            """Convert numpy arrays and types to JSON-serializable formats."""
            cleaned = {}
            for key, value in meta.items():
                if isinstance(value, (list, tuple)):
                    cleaned[key] = [str(v) if isinstance(v, (np.ndarray, np.generic)) else v for v in value]
                elif isinstance(value, (np.ndarray, np.generic)):
                    cleaned[key] = str(value)
                else:
                    cleaned[key] = value
            return cleaned

        metadata_clean = clean_metadata(metadata_dict)
        metadata_clean["file_name"] = file.filename or "audio.wav"
        metadata_clean["file_size"] = len(file_contents)
        metadata_clean["method"] = method

        # Create response
        response = FeatureExtractionResponse(
            success=True,
            features=features_list,
            labels=feature_labels,
            metadata=metadata_clean,
            method=method,
            message=f"Successfully extracted {method} features from {file.filename or 'uploaded audio'}"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during feature extraction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )
