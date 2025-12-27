"""Comprehensive tests for file operations in both local and remote modes."""
import json
import pytest
import hashlib
from pathlib import Path


class TestBasicFileOperations:
    """Tests for basic file upload operations."""

    def test_upload_single_file_local(self, local_experiment, sample_files, temp_project):
        """Test uploading a single file in local mode."""
        with local_experiment(name="file-test", project="test").run as experiment:
            result = experiment.files("models").upload(
                sample_files["model"],
                description="Model weights",
                tags=["model"]
            )

            assert result["filename"] == "model.txt"
            assert result["sizeBytes"] > 0
            assert "checksum" in result

        files_dir = temp_project / "test" / "file-test" / "files"
        assert files_dir.exists()
        saved_files = list(files_dir.glob("*/*/model.txt"))
        assert len(saved_files) == 1

    @pytest.mark.remote
    def test_upload_single_file_remote(self, remote_experiment, sample_files):
        """Test uploading a file in remote mode."""
        with remote_experiment(name="file-test-remote", project="test").run as experiment:
            result = experiment.files("models").upload(
                sample_files["model"],
                tags=["model", "remote"]
            )

            assert result["filename"] == "model.txt"

    def test_upload_multiple_files_local(self, local_experiment, sample_files, temp_project):
        """Test uploading multiple files."""
        with local_experiment(name="multi-file", project="test").run as experiment:
            experiment.files("models").upload(
                sample_files["model"])
            experiment.files("config").upload(
                sample_files["config"])
            experiment.files("results").upload(
                sample_files["results"])

        files_dir = temp_project / "test" / "multi-file" / "files"
        file_dirs = [d for d in files_dir.iterdir() if d.is_dir()]
        assert len(file_dirs) == 3

    @pytest.mark.remote
    def test_upload_multiple_files_remote(self, remote_experiment, sample_files):
        """Test uploading multiple files in remote mode."""
        with remote_experiment(name="multi-file-remote", project="test").run as experiment:
            experiment.files("models").upload(
                sample_files["model"])
            experiment.files("config").upload(
                sample_files["config"])


class TestFileMetadata:
    """Tests for file metadata and properties."""

    def test_file_with_metadata_local(self, local_experiment, sample_files, temp_project):
        """Test uploading file with custom metadata."""
        with local_experiment(name="file-meta", project="test").run as experiment:
            result = experiment.files("results").upload(
                sample_files["results"],
                description="Training results per epoch",
                tags=["results", "metrics"],
                metadata={"epochs": 10, "format": "csv"}
            )

            assert "uploadedAt" in result
            assert result["tags"] == ["results", "metrics"]

        metadata_file = temp_project / "test" / "file-meta" / "files" / ".files_metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            files_data = json.load(f)

        # Find our file
        our_file = next((f for f in files_data["files"] if f["filename"] == "results.csv"), None)
        assert our_file is not None
        assert our_file["description"] == "Training results per epoch"
        assert our_file["metadata"]["epochs"] == 10

    @pytest.mark.remote
    def test_file_with_metadata_remote(self, remote_experiment, sample_files):
        """Test file metadata in remote mode."""
        with remote_experiment(name="file-meta-remote", project="test").run as experiment:
            result = experiment.files("config").upload(
                sample_files["config"],
                description="Training configuration",
                tags=["config"],
                metadata={"version": "1.0"}
            )

            assert result["tags"] == ["config"]

    def test_file_checksum_local(self, local_experiment, sample_files):
        """Test that file checksum is correctly calculated."""
        with open(sample_files["model"], "rb") as f:
            expected_checksum = hashlib.sha256(f.read()).hexdigest()

        with local_experiment(name="file-checksum", project="test").run as experiment:
            result = experiment.files("models").upload(
                sample_files["model"])
            assert result["checksum"] == expected_checksum

    def test_file_size_metricing_local(self, local_experiment, sample_files):
        """Test that file sizes are correctly metriced."""
        model_size = Path(sample_files["model"]).stat().st_size

        with local_experiment(name="file-size", project="test").run as experiment:
            result = experiment.files("models").upload(
                sample_files["model"])
            assert result["sizeBytes"] == model_size

    def test_file_tags_local(self, local_experiment, sample_files, temp_project):
        """Test file tagging."""
        with local_experiment(name="file-tags", project="test").run as experiment:
            experiment.files("models").upload(
                sample_files["model"],
                tags=["best", "final", "v1.0", "production"]
            )

            files = experiment.files().list()

        assert len(files) == 1
        assert "best" in files[0]["tags"]
        assert "final" in files[0]["tags"]
        assert "production" in files[0]["tags"]


class TestListFiles:
    """Tests for listing uploaded files."""

    def test_list_files_local(self, local_experiment, sample_files):
        """Test listing all files in a experiment."""
        with local_experiment(name="file-list", project="test").run as experiment:
            experiment.files("models").upload(
                sample_files["model"])
            experiment.files("config").upload(
                sample_files["config"])

            files = experiment.files().list()

        assert len(files) == 2
        filenames = [f["filename"] for f in files]
        assert "model.txt" in filenames
        assert "config.json" in filenames

    @pytest.mark.remote
    def test_list_files_remote(self, remote_experiment, sample_files):
        """Test listing files in remote mode."""
        with remote_experiment(name="file-list-remote", project="test").run as experiment:
            experiment.files("models").upload(
                sample_files["model"])
            experiment.files("config").upload(
                sample_files["config"])

            files = experiment.files().list()
            assert len(files) >= 2

    def test_list_empty_files_local(self, local_experiment):
        """Test listing when no files uploaded."""
        with local_experiment(name="no-files", project="test").run as experiment:
            experiment.log("No files")
            # Depending on implementation, this may return empty list or handle gracefully


class TestFilePrefixes:
    """Tests for file path prefixes."""

    def test_file_prefixes_local(self, local_experiment, sample_files):
        """Test that file prefixes are correctly stored."""
        with local_experiment(name="file-prefix", project="test").run as experiment:
            experiment.files("models/v1").upload(
                sample_files["model"])
            experiment.files("configs/prod").upload(
                sample_files["config"])

            files = experiment.files().list()

        assert len(files) == 2
        paths = [f["path"] for f in files]
        assert "/models/v1" in paths
        assert "/configs/prod" in paths

    def test_nested_prefixes_local(self, local_experiment, sample_files):
        """Test deeply nested prefix paths."""
        with local_experiment(name="nested-prefix", project="test").run as experiment:
            experiment.files("a/b/c/d/e/models").upload(
                sample_files["model"]
            )

            files = experiment.files().list()

        assert len(files) == 1
        assert files[0]["path"] == "/a/b/c/d/e/models"

    def test_same_file_different_prefixes_local(self, local_experiment, sample_files):
        """Test uploading same file to different locations."""
        with local_experiment(name="same-file-diff-prefix", project="test").run as experiment:
            experiment.files("models/v1").upload(
                sample_files["model"])
            experiment.files("models/v2").upload(
                sample_files["model"])
            experiment.files("backup").upload(
                sample_files["model"])

            files = experiment.files().list()

        assert len(files) == 3
        assert all(f["filename"] == "model.txt" for f in files)
        paths = [f["path"] for f in files]
        assert "/models/v1" in paths
        assert "/models/v2" in paths
        assert "/backup" in paths


class TestFileTypes:
    """Tests for different file types."""

    def test_text_file_upload_local(self, local_experiment, sample_files, temp_project):
        """Test uploading text files."""
        with local_experiment(name="text-file", project="test").run as experiment:
            experiment.files("text").upload(
                sample_files["model"])

        files_dir = temp_project / "test" / "text-file" / "files"
        saved_files = list(files_dir.glob("*/*/model.txt"))
        assert len(saved_files) == 1

    def test_json_file_upload_local(self, local_experiment, sample_files):
        """Test uploading JSON files."""
        with local_experiment(name="json-file", project="test").run as experiment:
            result = experiment.files("json").upload(
                sample_files["config"])
            assert result["filename"] == "config.json"

    def test_csv_file_upload_local(self, local_experiment, sample_files):
        """Test uploading CSV files."""
        with local_experiment(name="csv-file", project="test").run as experiment:
            result = experiment.files("csv").upload(
                sample_files["results"])
            assert result["filename"] == "results.csv"

    def test_binary_file_upload_local(self, local_experiment, sample_files):
        """Test uploading binary files."""
        with local_experiment(name="binary-file", project="test").run as experiment:
            result = experiment.files("images").upload(
                sample_files["image"])
            assert result["filename"] == "test_image.png"
            assert result["sizeBytes"] > 0

    def test_large_file_upload_local(self, local_experiment, sample_files):
        """Test uploading larger files."""
        with local_experiment(name="large-file", project="test").run as experiment:
            result = experiment.files("large").upload(
                sample_files["large"])
            assert result["filename"] == "large_file.bin"
            assert result["sizeBytes"] == 1024 * 100  # 100 KB

    @pytest.mark.remote
    def test_various_file_types_remote(self, remote_experiment, sample_files):
        """Test uploading various file types in remote mode."""
        with remote_experiment(name="file-types-remote", project="test").run as experiment:
            experiment.files("text").upload(
                sample_files["model"])
            experiment.files("json").upload(
                sample_files["config"])
            experiment.files("images").upload(
                sample_files["image"])


class TestFileEdgeCases:
    """Tests for edge cases in file operations."""

    def test_file_with_spaces_in_name_local(self, local_experiment, tmp_path):
        """Test uploading file with spaces in filename."""
        file_with_spaces = tmp_path / "my file with spaces.txt"
        file_with_spaces.write_text("Content with spaces in filename")

        with local_experiment(name="spaces-file", project="test").run as experiment:
            result = experiment.files("files").upload(
                str(file_with_spaces))
            assert "my file with spaces.txt" in result["filename"]

    def test_file_with_unicode_name_local(self, local_experiment, tmp_path):
        """Test uploading file with unicode characters in name."""
        unicode_file = tmp_path / "文件_файл_αρχείο.txt"
        unicode_file.write_text("Unicode filename test")

        with local_experiment(name="unicode-file", project="test").run as experiment:
            result = experiment.files("files").upload(
                str(unicode_file))
            assert result["sizeBytes"] > 0

    def test_file_with_long_filename_local(self, local_experiment, tmp_path):
        """Test uploading file with very long filename."""
        long_name = "a" * 200 + ".txt"
        long_file = tmp_path / long_name
        long_file.write_text("Long filename test")

        with local_experiment(name="long-filename", project="test").run as experiment:
            result = experiment.files("files").upload(str(long_file))
            assert result["sizeBytes"] > 0

    def test_multiple_uploads_same_file_local(self, local_experiment, sample_files):
        """Test uploading the same file multiple times to same location."""
        with local_experiment(name="duplicate-upload", project="test").run as experiment:
            result1 = experiment.files("models").upload(
                sample_files["model"])
            result2 = experiment.files("models").upload(
                sample_files["model"])

            # Both uploads should succeed
            assert result1["filename"] == result2["filename"]

    def test_file_with_special_characters_local(self, local_experiment, tmp_path):
        """Test file with special characters in name."""
        special_file = tmp_path / "file-with_special.chars@123.txt"
        special_file.write_text("Special characters test")

        with local_experiment(name="special-chars-file", project="test").run as experiment:
            result = experiment.files("files").upload(
                str(special_file))
            assert result["sizeBytes"] > 0

    def test_empty_file_upload_local(self, local_experiment, tmp_path):
        """Test uploading an empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        with local_experiment(name="empty-file", project="test").run as experiment:
            result = experiment.files("files").upload(
                str(empty_file))
            assert result["sizeBytes"] == 0

    def test_file_with_long_metadata_local(self, local_experiment, sample_files):
        """Test file with extensive metadata."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(100)}

        with local_experiment(name="large-file-meta", project="test").run as experiment:
            result = experiment.files("models").upload(
                sample_files["model"],
                metadata=large_metadata
            )

            assert result["filename"] == "model.txt"

    def test_file_with_many_tags_local(self, local_experiment, sample_files):
        """Test file with many tags."""
        many_tags = [f"tag-{i}" for i in range(50)]

        with local_experiment(name="many-tags-file", project="test").run as experiment:
            result = experiment.files("models").upload(
                sample_files["model"],
                tags=many_tags
            )

            assert len(result["tags"]) == 50

    @pytest.mark.remote
    def test_large_file_remote(self, remote_experiment, sample_files):
        """Test uploading large file in remote mode."""
        with remote_experiment(name="large-file-remote", project="test").run as experiment:
            result = experiment.files("large").upload(
                sample_files["large"])
            assert result["sizeBytes"] == 1024 * 100


class TestFileOrganization:
    """Tests for file organization strategies."""

    def test_organize_by_type_local(self, local_experiment, sample_files):
        """Test organizing files by type."""
        with local_experiment(name="organized", project="test").run as experiment:
            experiment.files("models").upload(
                sample_files["model"], tags=["model"])
            experiment.files("configs").upload(
                sample_files["config"], tags=["config"])
            experiment.files("results").upload(
                sample_files["results"], tags=["results"])

            files = experiment.files().list()

        # Group by prefix
        models = [f for f in files if f["path"] == "/models"]
        configs = [f for f in files if f["path"] == "/configs"]
        results = [f for f in files if f["path"] == "/results"]

        assert len(models) == 1
        assert len(configs) == 1
        assert len(results) == 1

    def test_organize_by_version_local(self, local_experiment, sample_files):
        """Test organizing files by version."""
        with local_experiment(name="versioned", project="test").run as experiment:
            experiment.files("models/v1").upload(
                sample_files["model"], tags=["v1"])
            experiment.files("models/v2").upload(
                sample_files["model"], tags=["v2"])
            experiment.files("models/v3").upload(
                sample_files["model"], tags=["v3", "latest"])

            files = experiment.files().list()

        assert len(files) == 3
        versions = [f["path"] for f in files]
        assert "/models/v1" in versions
        assert "/models/v2" in versions
        assert "/models/v3" in versions

if __name__ == "__main__":
    """Run all tests with pytest."""
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
