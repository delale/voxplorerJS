"""
Pytest configuration and fixtures for API tests.

Provides:
- FastAPI TestClient
- Sample audio files (base64-encoded)
- Sample CSV/XLSX data
- Pre-computed feature data for reduction tests
"""

import io
import os
import base64
from pathlib import Path
from typing import List

import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app import app


@pytest.fixture
def client():
    """FastAPI TestClient for making requests to the API."""
    return TestClient(app)


@pytest.fixture
def sample_audio_file():
    """
    Load sample audio file and return as base64-encoded data URL.
    
    Uses sp01_sample1.wav from test/data/ directory.
    """
    audio_path = Path(__file__).parent.parent.parent / "test" / "data" / "sp01_sample1.wav"
    
    if not audio_path.exists():
        pytest.skip(f"Audio file not found: {audio_path}")
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    # Convert to base64 data URL
    b64_data = base64.b64encode(audio_bytes).decode("utf-8")
    data_url = f"data:audio/wav;base64,{b64_data}"
    
    return {
        "data_url": data_url,
        "bytes": audio_bytes,
        "filename": "sp01_sample1.wav",
    }


@pytest.fixture
def sample_audio_file_2():
    """
    Load second sample audio file (sp02_sample2.wav).
    """
    audio_path = Path(__file__).parent.parent.parent / "test" / "data" / "sp02_sample2.wav"
    
    if not audio_path.exists():
        pytest.skip(f"Audio file not found: {audio_path}")
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    b64_data = base64.b64encode(audio_bytes).decode("utf-8")
    data_url = f"data:audio/wav;base64,{b64_data}"
    
    return {
        "data_url": data_url,
        "bytes": audio_bytes,
        "filename": "sp02_sample2.wav",
    }


@pytest.fixture
def sample_csv_file():
    """
    Load sample CSV file and return as BytesIO.
    
    Uses fakeSaar.csv from test/data/ directory.
    """
    csv_path = Path(__file__).parent.parent.parent / "test" / "data" / "fakeSaar.csv"
    
    if not csv_path.exists():
        pytest.skip(f"CSV file not found: {csv_path}")
    
    with open(csv_path, "rb") as f:
        csv_bytes = f.read()
    
    return {
        "bytes": csv_bytes,
        "stream": io.BytesIO(csv_bytes),
        "filename": "fakeSaar.csv",
    }


@pytest.fixture
def sample_xlsx_file():
    """
    Load sample XLSX file and return as BytesIO.
    
    Uses fakeSaar.xlsx from test/data/ directory.
    """
    xlsx_path = Path(__file__).parent.parent.parent / "test" / "data" / "fakeSaar.xlsx"
    
    if not xlsx_path.exists():
        pytest.skip(f"XLSX file not found: {xlsx_path}")
    
    with open(xlsx_path, "rb") as f:
        xlsx_bytes = f.read()
    
    return {
        "bytes": xlsx_bytes,
        "stream": io.BytesIO(xlsx_bytes),
        "filename": "fakeSaar.xlsx",
    }


@pytest.fixture
def sample_tsv_file():
    """Create a sample TSV file in memory."""
    tsv_data = (
        "id\tname\tvalue1\tvalue2\n"
        "1\tA\t10.5\t20.3\n"
        "2\tB\t15.2\t25.1\n"
        "3\tC\t12.8\t22.9\n"
        "4\tD\t18.1\t27.4\n"
        "5\tE\t11.3\t21.6\n"
    )
    tsv_bytes = tsv_data.encode("utf-8")
    
    return {
        "bytes": tsv_bytes,
        "stream": io.BytesIO(tsv_bytes),
        "filename": "sample.tsv",
    }


@pytest.fixture
def sample_malformed_csv():
    """Create a malformed CSV file (inconsistent column count)."""
    csv_data = (
        "id,name,value\n"
        "1,A,10.5\n"
        "2,B,15.2,extra\n"  # Inconsistent column count
        "3,C\n"  # Missing columns
    )
    csv_bytes = csv_data.encode("utf-8")
    
    return {
        "bytes": csv_bytes,
        "stream": io.BytesIO(csv_bytes),
        "filename": "malformed.csv",
    }


@pytest.fixture
def empty_file():
    """Create an empty file."""
    return {
        "bytes": b"",
        "stream": io.BytesIO(b""),
        "filename": "empty.csv",
    }


@pytest.fixture
def txt_file():
    """Create a text file (unsupported format)."""
    txt_data = "This is a text file, not supported for parsing."
    txt_bytes = txt_data.encode("utf-8")
    
    return {
        "bytes": txt_bytes,
        "stream": io.BytesIO(txt_bytes),
        "filename": "unsupported.txt",
    }


@pytest.fixture
def sample_2d_data():
    """Generate sample 2D feature data (100 samples × 20 features)."""
    np.random.seed(42)
    return np.random.randn(100, 20).tolist()


@pytest.fixture
def sample_3d_data():
    """Generate sample 3D feature data (50 samples × 30 features)."""
    np.random.seed(123)
    return np.random.randn(50, 30).tolist()


@pytest.fixture
def sample_labels():
    """Generate sample labels for supervised learning (100 samples, 3 classes)."""
    return [i % 3 for i in range(100)]


@pytest.fixture
def sample_labels_3d():
    """Generate sample labels for 3D data (50 samples, 5 classes)."""
    return [i % 5 for i in range(50)]


@pytest.fixture
def synthetic_small_data():
    """Generate small synthetic data (10 samples × 5 features)."""
    np.random.seed(999)
    return np.random.randn(10, 5).tolist()


@pytest.fixture
def synthetic_small_labels():
    """Generate small synthetic labels (10 samples, 2 classes)."""
    return [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


@pytest.fixture
def large_feature_data():
    """Generate large feature data for stress testing (500 samples × 50 features)."""
    np.random.seed(555)
    return np.random.randn(500, 50).tolist()


@pytest.fixture
def metadata_audio_filename():
    """Audio filename with extractable metadata."""
    return "speaker_neutral_take1.wav"


@pytest.fixture
def metadata_variables():
    """Variables to extract from filename."""
    return "speaker,emotion,take"
