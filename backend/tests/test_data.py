"""
Tests for the data parsing API endpoint.

Tests cover:
- CSV parsing (standard and with different delimiters)
- TSV parsing
- XLSX parsing
- Error handling (unsupported formats, empty files, malformed data)
- Response schema validation
- Metadata extraction
- Row index column addition
"""

import io
import pytest
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


class TestDataParsingEndpoint:
    """Test suite for data parsing endpoint."""

    # Happy path: CSV parsing

    def test_csv_parsing_standard(self, sample_csv_file):
        """Test standard CSV parsing."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("fakeSaar.csv", sample_csv_file["bytes"], "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "metadata" in data
        assert len(data["data"]) > 0
        assert data["metadata"]["row_count"] > 0
        assert data["metadata"]["column_count"] > 0

    def test_csv_metadata_structure(self, sample_csv_file):
        """Test CSV parsing returns correct metadata."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("fakeSaar.csv", sample_csv_file["bytes"], "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()
        metadata = data["metadata"]

        # Check metadata fields
        assert "columns" in metadata
        assert "dtypes" in metadata
        assert "row_count" in metadata
        assert "column_count" in metadata
        assert "file_name" in metadata
        assert "file_size" in metadata

        # Verify types
        assert isinstance(metadata["columns"], list)
        assert isinstance(metadata["dtypes"], dict)
        assert isinstance(metadata["row_count"], int)
        assert isinstance(metadata["column_count"], int)
        assert isinstance(metadata["file_name"], str)
        assert isinstance(metadata["file_size"], int)

    def test_csv_data_structure(self, sample_csv_file):
        """Test CSV data is returned as list of dicts."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("fakeSaar.csv", sample_csv_file["bytes"], "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Check data is list of dicts
        assert isinstance(data["data"], list)
        for row in data["data"]:
            assert isinstance(row, dict)
            # Each row should have all columns
            assert len(row) > 0

    def test_csv_row_index_added(self, sample_csv_file):
        """Test that row_index column is added to data."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("fakeSaar.csv", sample_csv_file["bytes"], "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Check row_index column exists
        assert "row_index" in data["metadata"]["columns"]

        # Check row indices are sequential
        for i, row in enumerate(data["data"]):
            assert row["row_index"] == i

    def test_csv_dtypes_identified(self, sample_csv_file):
        """Test that data types are correctly identified."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("fakeSaar.csv", sample_csv_file["bytes"], "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()
        dtypes = data["metadata"]["dtypes"]

        # dtypes should map column names to type strings
        for col_name, col_type in dtypes.items():
            assert isinstance(col_name, str)
            assert isinstance(col_type, str)

    # Happy path: TSV parsing

    def test_tsv_parsing(self, sample_tsv_file):
        """Test TSV (tab-separated) parsing."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("sample.tsv", sample_tsv_file["bytes"], "text/tab-separated-values")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) > 0
        assert data["metadata"]["column_count"] > 0

    def test_tsv_data_correctness(self, sample_tsv_file):
        """Test that TSV data is parsed correctly."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("sample.tsv", sample_tsv_file["bytes"], "text/tab-separated-values")},
        )

        assert response.status_code == 200
        data = response.json()

        # Check column names
        assert "id" in data["metadata"]["columns"]
        assert "name" in data["metadata"]["columns"]
        assert "value1" in data["metadata"]["columns"]
        assert "value2" in data["metadata"]["columns"]

    # Happy path: XLSX parsing

    def test_xlsx_parsing(self, sample_xlsx_file):
        """Test XLSX (Excel) file parsing."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("fakeSaar.xlsx", sample_xlsx_file["bytes"], "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) > 0
        assert data["metadata"]["row_count"] > 0

    def test_xlsx_metadata_structure(self, sample_xlsx_file):
        """Test XLSX parsing returns correct metadata."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("fakeSaar.xlsx", sample_xlsx_file["bytes"], "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify metadata structure
        assert "metadata" in data
        assert data["metadata"]["file_name"] == "fakeSaar.xlsx"

    # Error cases: Unsupported formats

    def test_unsupported_file_format(self, txt_file):
        """Test error handling for unsupported file format."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("unsupported.txt", txt_file["bytes"], "text/plain")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "unsupported" in data["detail"].lower() or "format" in data["detail"].lower()

    def test_json_file_unsupported(self):
        """Test error handling for JSON file (unsupported)."""
        json_data = b'{"key": "value"}'
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("data.json", json_data, "application/json")},
        )

        assert response.status_code == 400

    # Error cases: Empty files

    def test_empty_csv_file(self, empty_file):
        """Test error handling for empty CSV file."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("empty.csv", empty_file["bytes"], "text/csv")},
        )

        assert response.status_code in [400, 422]
        data = response.json()
        assert "detail" in data
        assert "empty" in data["detail"].lower() or "no valid data" in data["detail"].lower()

    # Error cases: Malformed data

    def test_malformed_csv(self, sample_malformed_csv):
        """Test error handling for malformed CSV."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("malformed.csv", sample_malformed_csv["bytes"], "text/csv")},
        )

        # Polars may handle this gracefully or error - either is acceptable
        # Just ensure we get a response
        assert response.status_code in [200, 422]

    # Edge cases

    def test_csv_with_special_characters(self):
        """Test CSV parsing with special characters."""
        csv_data = "name,description\nAlice,Hello \"World\"\nBob,Test, with, commas\n"
        csv_bytes = csv_data.encode("utf-8")

        response = client.post(
            "/api/data/parse-table",
            files={"file": ("special.csv", csv_bytes, "text/csv")},
        )

        # Should handle special characters
        assert response.status_code in [200, 422]

    def test_csv_with_unicode(self):
        """Test CSV parsing with unicode characters."""
        csv_data = "name,value\n日本,100\nFrançais,200\n"
        csv_bytes = csv_data.encode("utf-8")

        response = client.post(
            "/api/data/parse-table",
            files={"file": ("unicode.csv", csv_bytes, "text/csv")},
        )

        assert response.status_code in [200, 422]

    def test_csv_with_null_values(self):
        """Test CSV parsing with missing/null values."""
        csv_data = "id,name,value\n1,Alice,100\n2,,200\n3,Bob,\n"
        csv_bytes = csv_data.encode("utf-8")

        response = client.post(
            "/api/data/parse-table",
            files={"file": ("nulls.csv", csv_bytes, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) > 0

    def test_csv_with_numeric_data(self):
        """Test CSV with numeric data is parsed correctly."""
        csv_data = "id,value1,value2\n1,10.5,20\n2,15.2,25\n3,12.8,22\n"
        csv_bytes = csv_data.encode("utf-8")

        response = client.post(
            "/api/data/parse-table",
            files={"file": ("numeric.csv", csv_bytes, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
        assert data["metadata"]["row_count"] == 3

    def test_large_csv_file(self):
        """Test parsing of large CSV file."""
        # Generate a large CSV
        rows = ["id,value1,value2,value3"]
        for i in range(1000):
            rows.append(f"{i},{i*1.5},{i*2.5},{i*3.5}")
        csv_data = "\n".join(rows)
        csv_bytes = csv_data.encode("utf-8")

        response = client.post(
            "/api/data/parse-table",
            files={"file": ("large.csv", csv_bytes, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["row_count"] == 1000

    def test_csv_with_mixed_types(self):
        """Test CSV with mixed column types."""
        csv_data = "id,name,score,active\n1,Alice,85.5,true\n2,Bob,92.3,false\n3,Charlie,78.1,true\n"
        csv_bytes = csv_data.encode("utf-8")

        response = client.post(
            "/api/data/parse-table",
            files={"file": ("mixed.csv", csv_bytes, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data["metadata"]["columns"]
        assert "name" in data["metadata"]["columns"]
        assert "score" in data["metadata"]["columns"]
        assert "active" in data["metadata"]["columns"]

    # Response schema validation

    def test_response_schema(self, sample_csv_file):
        """Test response matches DataParsingResponse schema."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("fakeSaar.csv", sample_csv_file["bytes"], "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Check all required fields
        assert "success" in data
        assert "data" in data
        assert "metadata" in data

        # Check types
        assert isinstance(data["success"], bool)
        assert isinstance(data["data"], list)
        assert isinstance(data["metadata"], dict)

    def test_root_data_endpoint(self):
        """Test the root data endpoint."""
        response = client.get("/api/data/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
        assert "/api/data/parse-table" in str(data["endpoints"])

    # Additional validation tests

    def test_column_count_matches_metadata(self, sample_csv_file):
        """Test that column count in metadata matches actual columns."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("fakeSaar.csv", sample_csv_file["bytes"], "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Column count should match length of columns list
        assert data["metadata"]["column_count"] == len(data["metadata"]["columns"])

    def test_row_count_matches_data_length(self, sample_csv_file):
        """Test that row count in metadata matches actual data."""
        response = client.post(
            "/api/data/parse-table",
            files={"file": ("fakeSaar.csv", sample_csv_file["bytes"], "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        # Row count should match data length
        assert data["metadata"]["row_count"] == len(data["data"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
