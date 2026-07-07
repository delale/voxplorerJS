"""
Tests for the dimensionality reduction API endpoint.

Tests cover:
- All four algorithms: PCA, UMAP, t-SNE, MDS
- 2D and 3D output
- Supervised mode for UMAP
- Error handling (empty data, invalid algorithms, etc.)
- Response format and data types
- Correct dimensionality of output
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


class TestReductionEndpoint:
    """Test suite for dimensionality reduction endpoint."""

    @pytest.fixture
    def sample_2d_data(self):
        """Generate sample 2D feature data (100 samples × 20 features)."""
        np.random.seed(42)
        return np.random.randn(100, 20).tolist()

    @pytest.fixture
    def sample_labels(self):
        """Generate sample labels for supervised learning."""
        return [i % 3 for i in range(100)]

    def test_pca_2d_reduction(self, sample_2d_data):
        """Test PCA reduction to 2D."""
        payload = {
            "data": sample_2d_data,
            "algorithm": "pca",
            "n_components": 2,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["algorithm_used"] == "pca"
        assert data["n_components"] == 2
        assert data["n_samples"] == 100
        assert data["n_features_input"] == 20
        assert len(data["reduced_data"]) == 2
        assert "DIM1" in data["reduced_data"]
        assert "DIM2" in data["reduced_data"]
        assert len(data["reduced_data"]["DIM1"]) == 100
        assert len(data["reduced_data"]["DIM2"]) == 100
        assert data["explained_variance"] is not None
        assert len(data["explained_variance"]) == 2

    def test_pca_3d_reduction(self, sample_2d_data):
        """Test PCA reduction to 3D."""
        payload = {
            "data": sample_2d_data,
            "algorithm": "pca",
            "n_components": 3,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["n_components"] == 3
        assert len(data["reduced_data"]) == 3
        assert "DIM1" in data["reduced_data"]
        assert "DIM2" in data["reduced_data"]
        assert "DIM3" in data["reduced_data"]
        assert len(data["reduced_data"]["DIM3"]) == 100

    def test_umap_2d_reduction(self, sample_2d_data):
        """Test UMAP reduction to 2D."""
        payload = {
            "data": sample_2d_data,
            "algorithm": "umap",
            "n_components": 2,
            "n_neighbors": 15,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["algorithm_used"] == "umap"
        assert data["n_components"] == 2
        assert len(data["reduced_data"]) == 2
        assert data["explained_variance"] is None

    def test_umap_supervised(self, sample_2d_data, sample_labels):
        """Test UMAP with supervised mode."""
        payload = {
            "data": sample_2d_data,
            "algorithm": "umap",
            "n_components": 2,
            "supervised": True,
            "y": sample_labels,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["reduced_data"]) == 2

    def test_tsne_2d_reduction(self, sample_2d_data):
        """Test t-SNE reduction to 2D."""
        payload = {
            "data": sample_2d_data,
            "algorithm": "tsne",
            "n_components": 2,
            "perplexity": 30.0,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["algorithm_used"] == "tsne"
        assert data["n_components"] == 2
        assert len(data["reduced_data"]) == 2

    def test_mds_2d_reduction(self, sample_2d_data):
        """Test MDS reduction to 2D."""
        payload = {
            "data": sample_2d_data,
            "algorithm": "mds",
            "n_components": 2,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["algorithm_used"] == "mds"
        assert data["n_components"] == 2
        assert len(data["reduced_data"]) == 2

    def test_all_algorithms(self, sample_2d_data):
        """Test all four algorithms work correctly."""
        for algorithm in ["pca", "umap", "tsne", "mds"]:
            payload = {
                "data": sample_2d_data,
                "algorithm": algorithm,
                "n_components": 2,
            }
            response = client.post("/api/reduction/reduce", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["algorithm_used"] == algorithm

    # Error cases

    def test_empty_data(self):
        """Test error handling for empty data."""
        payload = {
            "data": [],
            "algorithm": "pca",
            "n_components": 2,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 422
        assert "empty" in response.json()["detail"].lower()

    def test_single_sample(self, sample_2d_data):
        """Test error handling for single sample."""
        payload = {
            "data": [sample_2d_data[0]],
            "algorithm": "pca",
            "n_components": 2,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 422

    def test_invalid_algorithm(self, sample_2d_data):
        """Test error handling for invalid algorithm."""
        payload = {
            "data": sample_2d_data,
            "algorithm": "invalid_algo",
            "n_components": 2,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 422

    def test_supervised_on_non_umap(self, sample_2d_data, sample_labels):
        """Test error when supervised=True on non-UMAP algorithm."""
        for algorithm in ["pca", "tsne", "mds"]:
            payload = {
                "data": sample_2d_data,
                "algorithm": algorithm,
                "n_components": 2,
                "supervised": True,
                "y": sample_labels,
            }
            response = client.post("/api/reduction/reduce", json=payload)
            
            assert response.status_code == 422
            assert "supervised" in response.json()["detail"].lower()

    def test_supervised_without_labels(self, sample_2d_data):
        """Test error when supervised=True but no labels provided."""
        payload = {
            "data": sample_2d_data,
            "algorithm": "umap",
            "n_components": 2,
            "supervised": True,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 422
        assert "label" in response.json()["detail"].lower()

    def test_mismatched_labels_length(self, sample_2d_data):
        """Test error when label count doesn't match sample count."""
        payload = {
            "data": sample_2d_data,
            "algorithm": "umap",
            "n_components": 2,
            "supervised": True,
            "y": [0, 1],  # Only 2 labels instead of 100
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 422
        assert "length" in response.json()["detail"].lower()

    def test_invalid_n_components_zero(self, sample_2d_data):
        """Test error when n_components is 0."""
        payload = {
            "data": sample_2d_data,
            "algorithm": "pca",
            "n_components": 0,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 422

    def test_invalid_n_components_too_high(self, sample_2d_data):
        """Test error when n_components is too high."""
        payload = {
            "data": sample_2d_data,
            "algorithm": "pca",
            "n_components": 4,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 422

    def test_invalid_data_type(self):
        """Test error when data is not numeric."""
        payload = {
            "data": [["a", "b"], ["c", "d"]],
            "algorithm": "pca",
            "n_components": 2,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 422

    def test_1d_data(self):
        """Test error when data is 1D instead of 2D."""
        payload = {
            "data": [1, 2, 3, 4],
            "algorithm": "pca",
            "n_components": 2,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 422

    # Edge cases

    def test_small_dataset(self):
        """Test with minimal dataset (2 samples)."""
        payload = {
            "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "algorithm": "pca",
            "n_components": 2,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["n_samples"] == 2

    def test_single_feature(self):
        """Test with single feature (1D features)."""
        payload = {
            "data": [[1.0], [2.0], [3.0], [4.0]],
            "algorithm": "pca",
            "n_components": 1,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["n_features_input"] == 1

    def test_response_format(self, sample_2d_data):
        """Test response format matches schema."""
        payload = {
            "data": sample_2d_data,
            "algorithm": "pca",
            "n_components": 2,
        }
        response = client.post("/api/reduction/reduce", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields exist
        assert "success" in data
        assert "algorithm_used" in data
        assert "n_components" in data
        assert "reduced_data" in data
        assert "n_samples" in data
        assert "n_features_input" in data
        assert "message" in data
        
        # Check types
        assert isinstance(data["success"], bool)
        assert isinstance(data["algorithm_used"], str)
        assert isinstance(data["n_components"], int)
        assert isinstance(data["reduced_data"], dict)
        assert isinstance(data["n_samples"], int)
        assert isinstance(data["n_features_input"], int)

    def test_seed_reproducibility(self, sample_2d_data):
        """Test that same seed produces same results."""
        payload1 = {
            "data": sample_2d_data,
            "algorithm": "umap",
            "n_components": 2,
            "seed": 42,
        }
        payload2 = {
            "data": sample_2d_data,
            "algorithm": "umap",
            "n_components": 2,
            "seed": 42,
        }
        
        response1 = client.post("/api/reduction/reduce", json=payload1)
        response2 = client.post("/api/reduction/reduce", json=payload2)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Results should be identical with same seed
        dim1_1 = data1["reduced_data"]["DIM1"]
        dim1_2 = data2["reduced_data"]["DIM1"]
        
        for v1, v2 in zip(dim1_1, dim1_2):
            assert abs(v1 - v2) < 1e-6

    def test_root_endpoint(self):
        """Test the root reduction endpoint."""
        response = client.get("/api/reduction/")
        
        assert response.status_code == 200
        data = response.json()
        assert "available_algorithms" in data
        assert "endpoints" in data
        assert data["available_algorithms"] == ["pca", "umap", "tsne", "mds"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
