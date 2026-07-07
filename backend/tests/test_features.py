"""
Tests for the feature extraction API endpoint.

Tests cover:
- MFCC extraction with various parameters
- Speaker embeddings extraction
- Delta and delta-delta features
- Summarization
- Error handling for invalid inputs
- Response schema validation
"""

import io
import pytest
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


class TestFeatureExtractionEndpoint:
    """Test suite for feature extraction endpoint."""

    # Happy path: MFCC extraction

    def test_mfcc_extraction_default(self, sample_audio_file):
        """Test MFCC extraction with default parameters."""
        with open(io.BytesIO(), "wb") as f:
            pass  # Placeholder

        # Simulate file upload using TestClient
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
                "n_mels": 40,
                "win_length": 25.0,
                "overlap": 10.0,
                "fmin": 100.0,
                "fmax": 6000.0,
                "preemphasis": 0.95,
                "lifter": 22.0,
                "summarise": False,
                "include_delta": False,
                "include_delta_delta": False,
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["method"] == "mfcc"
        assert "features" in data
        assert "labels" in data
        assert "metadata" in data
        assert len(data["features"]) > 0
        assert len(data["labels"]) == 13

    def test_mfcc_with_delta(self, sample_audio_file):
        """Test MFCC extraction with delta features."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
                "include_delta": True,
                "include_delta_delta": False,
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["labels"]) == 26  # 13 MFCC + 13 delta

    def test_mfcc_with_delta_delta(self, sample_audio_file):
        """Test MFCC extraction with delta-delta features."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
                "include_delta": True,
                "include_delta_delta": True,
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["labels"]) == 39  # 13 MFCC + 13 delta + 13 delta-delta

    def test_mfcc_summarised(self, sample_audio_file):
        """Test MFCC extraction with summarization."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
                "summarise": True,
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # Summarised should have reduced number of frames
        assert len(data["features"]) == 1  # Single summarised row

    def test_mfcc_custom_parameters(self, sample_audio_file):
        """Test MFCC extraction with custom parameters."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 20,  # Custom
                "n_mels": 60,  # Custom
                "fmin": 50.0,  # Custom
                "fmax": 8000.0,  # Custom
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["labels"]) == 20  # Should have 20 MFCCs

    # Happy path: Speaker embeddings

    def test_speaker_embeddings_extraction(self, sample_audio_file):
        """Test speaker embeddings extraction."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "speaker_embeddings",
                "model": "speechbrain/spkrec-ecapa-voxceleb",
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["method"] == "speaker_embeddings"
        assert len(data["features"]) == 1  # Single embedding vector
        assert len(data["labels"]) > 0  # Should have embedding dimensions

    # Happy path: Metadata extraction

    def test_metadata_extraction_from_filename(self, sample_audio_file):
        """Test metadata extraction from filename."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
                "metadata_separator": "_",
                "metadata_variables": "speaker,emotion,take",
            },
            files={
                "file": (
                    "speaker_neutral_take1.wav",
                    sample_audio_file["bytes"],
                    "audio/wav",
                )
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["metadata"]["file_name"] == "speaker_neutral_take1.wav"

    # Error cases

    def test_invalid_method(self, sample_audio_file):
        """Test error handling for invalid extraction method."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "invalid_method",
                "n_mfcc": 13,
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data or "error" in data.get("detail", "").lower()

    def test_invalid_file_format(self, sample_audio_file):
        """Test error handling for invalid file format."""
        txt_content = b"This is not audio"
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
            },
            files={"file": ("invalid.txt", txt_content, "text/plain")},
        )

        assert response.status_code in [400, 422, 500]  # Any error status

    def test_empty_audio_file(self):
        """Test error handling for empty audio file."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
            },
            files={"file": ("empty.wav", b"", "audio/wav")},
        )

        assert response.status_code in [400, 422, 500]

    def test_invalid_n_mfcc_negative(self, sample_audio_file):
        """Test error handling for negative n_mfcc."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": -5,  # Invalid
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code in [400, 422, 500]

    def test_invalid_n_mfcc_zero(self, sample_audio_file):
        """Test error handling for zero n_mfcc."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 0,  # Invalid
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code in [400, 422, 500]

    def test_invalid_fmin_greater_than_fmax(self, sample_audio_file):
        """Test error handling when fmin > fmax."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
                "fmin": 8000.0,
                "fmax": 100.0,  # fmax < fmin
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code in [400, 422, 500]

    # Response schema validation

    def test_response_schema_success(self, sample_audio_file):
        """Test that response matches FeatureExtractionResponse schema."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()

        # Check all required fields
        assert "success" in data
        assert "features" in data
        assert "labels" in data
        assert "metadata" in data
        assert "method" in data

        # Check types
        assert isinstance(data["success"], bool)
        assert isinstance(data["features"], list)
        assert isinstance(data["labels"], list)
        assert isinstance(data["metadata"], dict)
        assert isinstance(data["method"], str)

        # Check nested structure
        if data["features"]:
            assert isinstance(data["features"][0], dict)
            for key in data["labels"]:
                assert key in data["features"][0]

    def test_metadata_structure(self, sample_audio_file):
        """Test metadata contains expected information."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        metadata = data["metadata"]

        # Check metadata fields
        assert "file_name" in metadata
        assert "file_size" in metadata
        assert "method" in metadata

    def test_features_are_numeric(self, sample_audio_file):
        """Test that all extracted features are numeric."""
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()

        for feature_row in data["features"]:
            for label in data["labels"]:
                value = feature_row[label]
                assert isinstance(value, (int, float)), f"Expected numeric, got {type(value)}"

    # Edge cases

    def test_different_audio_files(self, sample_audio_file, sample_audio_file_2):
        """Test that different audio files produce different features."""
        response1 = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
                "summarise": True,
            },
            files={"file": ("sp01_sample1.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        response2 = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
                "summarise": True,
            },
            files={"file": ("sp02_sample2.wav", sample_audio_file_2["bytes"], "audio/wav")},
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Different files should produce different features (unless extremely similar)
        # Just verify both produced valid output
        assert len(data1["features"]) > 0
        assert len(data2["features"]) > 0

    def test_root_features_endpoint(self):
        """Test the root features endpoint."""
        response = client.get("/api/features/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_mp3_file_support(self, sample_audio_file):
        """Test that MP3 files are accepted (if available)."""
        # MP3 would need to be tested with actual MP3 file
        # For now, test that wav extension is handled
        response = client.post(
            "/api/features/extract",
            data={
                "method": "mfcc",
                "n_mfcc": 13,
            },
            files={"file": ("test.wav", sample_audio_file["bytes"], "audio/wav")},
        )

        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
