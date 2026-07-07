"""
Data parsing and loading endpoints.

Provides endpoints for:
- Parsing CSV/TSV/XLSX files
- Loading tabular data
- Data validation
"""

import logging
import os
import io
from typing import Any

from fastapi import APIRouter, UploadFile, File, HTTPException, status
import polars as pl

from backend.schemas.requests import DataParsingResponse, DataParsingMetadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])

ALLOWED_FORMATS = {".csv", ".tsv", ".xlsx", ".xls", ".xlsb"}


def parse_table_contents(
    file_bytes: bytes, filename: str
) -> tuple[list[dict[str, Any]], pl.DataFrame]:
    """
    Parse table file contents and return data as list of dicts.

    Args:
        file_bytes: Raw file bytes
        filename: Original filename for extension detection

    Returns:
        Tuple of (data_dicts, dataframe) where data_dicts is JSON-serializable

    Raises:
        ValueError: If file format is unsupported or parsing fails
    """
    file_stream = io.BytesIO(file_bytes)
    ext = os.path.splitext(filename)[-1].lower()

    try:
        if ext == ".csv":
            data_table = pl.read_csv(file_stream)
        elif ext == ".tsv":
            data_table = pl.read_csv(file_stream, separator="\t")
        elif ext in (".xls", ".xlsb", ".xlsx"):
            data_table = pl.read_excel(file_stream)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported formats: {', '.join(ALLOWED_FORMATS)}"
            )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse {filename}: {str(e)}")

    if data_table.is_empty():
        raise ValueError("File is empty or contains no valid data")

    # Add row_index if not already present
    if "row_index" not in data_table.columns:
        data_table = data_table.with_row_index("row_index")

    # Convert to JSON-serializable format
    data_dicts = data_table.to_dicts()

    return data_dicts, data_table


def extract_metadata(
    data_table: pl.DataFrame, filename: str, file_size: int
) -> dict[str, Any]:
    """
    Extract metadata from parsed table.

    Args:
        data_table: Polars DataFrame
        filename: Original filename
        file_size: File size in bytes

    Returns:
        Dictionary with metadata
    """
    # Get data types as strings
    dtypes = {col: str(data_table[col].dtype) for col in data_table.columns}

    return {
        "columns": data_table.columns,
        "dtypes": dtypes,
        "row_count": data_table.height,
        "column_count": data_table.width,
        "file_name": filename,
        "file_size": file_size,
    }


@router.get("/")
async def data_root():
    """Root endpoint for data parsing endpoints."""
    return {
        "message": "Data parsing endpoints",
        "endpoints": {
            "parse-table": "POST /api/data/parse-table",
        },
    }


@router.post("/parse-table", response_model=DataParsingResponse)
async def parse_table(
    file: UploadFile = File(...),
) -> DataParsingResponse:
    """
    Parse uploaded table file (CSV/TSV/XLSX).

    Reads the uploaded file, detects format from extension, and parses
    the contents into a structured format. Automatically adds row_index
    column if not present.

    Args:
        file: Uploaded file (CSV, TSV, or XLSX format)

    Returns:
        DataParsingResponse with parsed data and metadata

    Raises:
        HTTPException: 400 for unsupported formats, 422 for malformed data, 500 for server errors
    """
    logger.info(f"Parsing table file: {file.filename}")

    try:
        # Read file contents
        file_bytes = await file.read()
        file_size = len(file_bytes)

        if file_size == 0:
            raise ValueError("Uploaded file is empty")

        # Parse table contents
        data_dicts, data_table = parse_table_contents(file_bytes, file.filename)

        # Extract metadata
        metadata = extract_metadata(data_table, file.filename, file_size)

        logger.info(
            f"Successfully parsed {file.filename}: "
            f"{metadata['row_count']} rows × {metadata['column_count']} columns"
        )

        return DataParsingResponse(
            success=True,
            data=data_dicts,
            metadata=DataParsingMetadata(**metadata),
            message=f"Successfully parsed {file.filename}",
        )

    except ValueError as e:
        error_msg = str(e)
        logger.warning(f"Validation error parsing {file.filename}: {error_msg}")

        if "Unsupported file format" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_msg,
            )

    except Exception as e:
        error_msg = f"Unexpected error parsing {file.filename}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while parsing file",
        )
