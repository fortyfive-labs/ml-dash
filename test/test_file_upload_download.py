"""Comprehensive tests for file upload and download operations."""
import json
import pytest
import hashlib
from pathlib import Path


class TestFileUploadDownload:
    """Tests for file upload and download workflow."""

    def test_upload_and_download_text_file_local(self, local_experiment, sample_files, tmp_path):
        """Test uploading and downloading a text file in local mode."""
        with local_experiment(name="upload-download", project="test").run as experiment:
            # Upload file
            upload_result = experiment.files(
                file_path=sample_files["model"],
                prefix="/models",
                description="Model file"
            ).save()

            file_id = upload_result["id"]
            original_checksum = upload_result["checksum"]

            # Download file
            download_path = tmp_path / "downloaded_model.txt"
            downloaded = experiment.files(
                file_id=file_id,
                dest_path=str(download_path)
            ).download()

        # Verify download
        assert Path(downloaded).exists()
        assert download_path.read_text() == Path(sample_files["model"]).read_text()

        # Verify checksum
        with open(downloaded, "rb") as f:
            downloaded_checksum = hashlib.sha256(f.read()).hexdigest()
        assert downloaded_checksum == original_checksum

    @pytest.mark.remote
    def test_upload_and_download_text_file_remote(self, remote_experiment, sample_files, tmp_path):
        """Test uploading and downloading a text file in remote mode."""
        with remote_experiment(name="upload-download-remote", project="test").run as experiment:
            # Upload file
            upload_result = experiment.files(
                file_path=sample_files["model"],
                prefix="/models"
            ).save()

            file_id = upload_result["id"]
            original_checksum = upload_result["checksum"]

            # Download file
            download_path = tmp_path / "downloaded_model_remote.txt"
            downloaded = experiment.files(
                file_id=file_id,
                dest_path=str(download_path)
            ).download()

        # Verify download
        assert Path(downloaded).exists()
        assert download_path.read_text() == Path(sample_files["model"]).read_text()

    def test_upload_and_download_json_file_local(self, local_experiment, sample_files, tmp_path):
        """Test uploading and downloading a JSON file."""
        with local_experiment(name="json-upload-download", project="test").run as experiment:
            # Upload JSON file
            upload_result = experiment.files(
                file_path=sample_files["config"],
                prefix="/configs",
                tags=["config", "json"]
            ).save()

            # Download it back
            download_path = tmp_path / "downloaded_config.json"
            downloaded = experiment.files(
                file_id=upload_result["id"],
                dest_path=str(download_path)
            ).download()

        # Verify content matches
        original_content = json.loads(Path(sample_files["config"]).read_text())
        downloaded_content = json.loads(Path(downloaded).read_text())
        assert original_content == downloaded_content

    def test_upload_and_download_binary_file_local(self, local_experiment, sample_files, tmp_path):
        """Test uploading and downloading a binary file."""
        with local_experiment(name="binary-upload-download", project="test").run as experiment:
            # Upload binary image file
            upload_result = experiment.files(
                file_path=sample_files["image"],
                prefix="/images",
                tags=["image", "png"]
            ).save()

            original_size = upload_result["sizeBytes"]

            # Download it back
            download_path = tmp_path / "downloaded_image.png"
            downloaded = experiment.files(
                file_id=upload_result["id"],
                dest_path=str(download_path)
            ).download()

        # Verify binary content matches
        original_bytes = Path(sample_files["image"]).read_bytes()
        downloaded_bytes = Path(downloaded).read_bytes()
        assert original_bytes == downloaded_bytes
        assert len(downloaded_bytes) == original_size

    def test_upload_and_download_large_file_local(self, local_experiment, sample_files, tmp_path):
        """Test uploading and downloading a large file."""
        with local_experiment(name="large-upload-download", project="test").run as experiment:
            # Upload large file (100 KB)
            upload_result = experiment.files(
                file_path=sample_files["large"],
                prefix="/large",
                description="Large binary file"
            ).save()

            assert upload_result["sizeBytes"] == 1024 * 100

            # Download it back
            download_path = tmp_path / "downloaded_large.bin"
            downloaded = experiment.files(
                file_id=upload_result["id"],
                dest_path=str(download_path)
            ).download()

        # Verify size and checksum
        assert Path(downloaded).stat().st_size == 1024 * 100

        original_bytes = Path(sample_files["large"]).read_bytes()
        downloaded_bytes = Path(downloaded).read_bytes()
        assert original_bytes == downloaded_bytes

    def test_download_to_current_directory_local(self, local_experiment, sample_files, tmp_path, monkeypatch):
        """Test downloading file without specifying dest_path (uses original filename)."""
        # Change to tmp directory
        monkeypatch.chdir(tmp_path)

        with local_experiment(name="download-current-dir", project="test").run as experiment:
            # Upload file
            upload_result = experiment.files(
                file_path=sample_files["model"],
                prefix="/models"
            ).save()

            # Download without dest_path (should use original filename in current directory)
            downloaded = experiment.files(file_id=upload_result["id"]).download()

        # Verify file is in current directory with original name
        assert Path(downloaded).exists()
        assert Path(downloaded).name == "model.txt"
        # Verify the file is in the tmp_path directory
        assert (tmp_path / Path(downloaded).name).exists()

    def test_upload_download_multiple_files_local(self, local_experiment, sample_files, tmp_path):
        """Test uploading and downloading multiple files."""
        with local_experiment(name="multi-upload-download", project="test").run as experiment:
            # Upload multiple files
            model_result = experiment.files(
                file_path=sample_files["model"],
                prefix="/models"
            ).save()

            config_result = experiment.files(
                file_path=sample_files["config"],
                prefix="/configs"
            ).save()

            results_result = experiment.files(
                file_path=sample_files["results"],
                prefix="/results"
            ).save()

            # Download all files
            model_download = experiment.files(
                file_id=model_result["id"],
                dest_path=str(tmp_path / "model.txt")
            ).download()

            config_download = experiment.files(
                file_id=config_result["id"],
                dest_path=str(tmp_path / "config.json")
            ).download()

            results_download = experiment.files(
                file_id=results_result["id"],
                dest_path=str(tmp_path / "results.csv")
            ).download()

        # Verify all downloads
        assert Path(model_download).exists()
        assert Path(config_download).exists()
        assert Path(results_download).exists()

        # Verify content
        assert Path(model_download).read_text() == Path(sample_files["model"]).read_text()
        assert Path(config_download).read_text() == Path(sample_files["config"]).read_text()
        assert Path(results_download).read_text() == Path(sample_files["results"]).read_text()

    def test_upload_download_with_metadata_local(self, local_experiment, sample_files, tmp_path):
        """Test that metadata is preserved after download."""
        with local_experiment(name="metadata-upload-download", project="test").run as experiment:
            # Upload with metadata
            upload_result = experiment.files(
                file_path=sample_files["model"],
                prefix="/models",
                description="Test model with metadata",
                tags=["test", "model", "v1"],
                metadata={"version": "1.0", "accuracy": 0.95}
            ).save()

            # List files to verify metadata
            files = experiment.files().list()
            our_file = next(f for f in files if f["id"] == upload_result["id"])

            assert our_file["description"] == "Test model with metadata"
            assert set(our_file["tags"]) == {"test", "model", "v1"}
            assert our_file["metadata"]["version"] == "1.0"
            assert our_file["metadata"]["accuracy"] == 0.95

            # Download file
            download_path = tmp_path / "model_with_metadata.txt"
            downloaded = experiment.files(
                file_id=upload_result["id"],
                dest_path=str(download_path)
            ).download()

        # Verify file downloaded correctly (content should be same regardless of metadata)
        assert Path(downloaded).read_text() == Path(sample_files["model"]).read_text()

    def test_upload_download_checksum_verification_local(self, local_experiment, sample_files, tmp_path):
        """Test that checksum is verified during download."""
        with local_experiment(name="checksum-verify", project="test").run as experiment:
            # Upload file
            upload_result = experiment.files(
                file_path=sample_files["model"],
                prefix="/models"
            ).save()

            original_checksum = upload_result["checksum"]

            # Download file
            download_path = tmp_path / "verified_model.txt"
            downloaded = experiment.files(
                file_id=upload_result["id"],
                dest_path=str(download_path)
            ).download()

        # Compute checksum of downloaded file
        with open(downloaded, "rb") as f:
            downloaded_checksum = hashlib.sha256(f.read()).hexdigest()

        # Checksums should match
        assert downloaded_checksum == original_checksum

    @pytest.mark.remote
    def test_upload_download_multiple_files_remote(self, remote_experiment, sample_files, tmp_path):
        """Test uploading and downloading multiple files in remote mode."""
        with remote_experiment(name="multi-remote-download", project="test").run as experiment:
            # Upload files
            model_result = experiment.files(
                file_path=sample_files["model"],
                prefix="/models"
            ).save()

            config_result = experiment.files(
                file_path=sample_files["config"],
                prefix="/configs"
            ).save()

            # Download files
            model_download = experiment.files(
                file_id=model_result["id"],
                dest_path=str(tmp_path / "remote_model.txt")
            ).download()

            config_download = experiment.files(
                file_id=config_result["id"],
                dest_path=str(tmp_path / "remote_config.json")
            ).download()

        # Verify downloads
        assert Path(model_download).exists()
        assert Path(config_download).exists()

    def test_upload_same_file_to_different_prefixes_and_download_local(self, local_experiment, sample_files, tmp_path):
        """Test uploading same file to different locations and downloading each."""
        with local_experiment(name="same-file-diff-locations", project="test").run as experiment:
            # Upload same file to different prefixes
            v1_result = experiment.files(
                file_path=sample_files["model"],
                prefix="/models/v1"
            ).save()

            v2_result = experiment.files(
                file_path=sample_files["model"],
                prefix="/models/v2"
            ).save()

            backup_result = experiment.files(
                file_path=sample_files["model"],
                prefix="/backup"
            ).save()

            # Download all versions
            v1_download = experiment.files(
                file_id=v1_result["id"],
                dest_path=str(tmp_path / "model_v1.txt")
            ).download()

            v2_download = experiment.files(
                file_id=v2_result["id"],
                dest_path=str(tmp_path / "model_v2.txt")
            ).download()

            backup_download = experiment.files(
                file_id=backup_result["id"],
                dest_path=str(tmp_path / "model_backup.txt")
            ).download()

        # All downloads should have same content
        original_content = Path(sample_files["model"]).read_text()
        assert Path(v1_download).read_text() == original_content
        assert Path(v2_download).read_text() == original_content
        assert Path(backup_download).read_text() == original_content

    def test_download_nonexistent_file_local(self, local_experiment):
        """Test downloading a file that doesn't exist."""
        with local_experiment(name="download-nonexistent", project="test").run as experiment:
            # Try to download file with fake ID
            with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
                experiment.files(file_id="nonexistent-id-12345").download()

    def test_upload_and_re_download_local(self, local_experiment, sample_files, tmp_path):
        """Test uploading, downloading, and re-downloading the same file."""
        with local_experiment(name="re-download", project="test").run as experiment:
            # Upload
            upload_result = experiment.files(
                file_path=sample_files["model"],
                prefix="/models"
            ).save()

            file_id = upload_result["id"]

            # First download
            first_download = experiment.files(
                file_id=file_id,
                dest_path=str(tmp_path / "first_download.txt")
            ).download()

            # Second download (same file)
            second_download = experiment.files(
                file_id=file_id,
                dest_path=str(tmp_path / "second_download.txt")
            ).download()

        # Both downloads should have identical content
        assert Path(first_download).read_text() == Path(second_download).read_text()
        assert Path(first_download).read_text() == Path(sample_files["model"]).read_text()


class TestFileUploadDownloadEdgeCases:
    """Edge case tests for file upload and download."""

    def test_download_overwrites_existing_file_local(self, local_experiment, sample_files, tmp_path):
        """Test that download overwrites existing file at dest_path."""
        download_path = tmp_path / "existing_file.txt"
        download_path.write_text("This will be overwritten")

        with local_experiment(name="overwrite-download", project="test").run as experiment:
            # Upload file
            upload_result = experiment.files(
                file_path=sample_files["model"],
                prefix="/models"
            ).save()

            # Download to existing file location
            downloaded = experiment.files(
                file_id=upload_result["id"],
                dest_path=str(download_path)
            ).download()

        # Verify original content was overwritten
        assert Path(downloaded).read_text() == Path(sample_files["model"]).read_text()
        assert Path(downloaded).read_text() != "This will be overwritten"

    def test_upload_download_empty_file_local(self, local_experiment, tmp_path):
        """Test uploading and downloading an empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        with local_experiment(name="empty-file-download", project="test").run as experiment:
            # Upload empty file
            upload_result = experiment.files(
                file_path=str(empty_file),
                prefix="/files"
            ).save()

            assert upload_result["sizeBytes"] == 0

            # Download it back
            download_path = tmp_path / "downloaded_empty.txt"
            downloaded = experiment.files(
                file_id=upload_result["id"],
                dest_path=str(download_path)
            ).download()

        # Verify downloaded file is also empty
        assert Path(downloaded).read_text() == ""
        assert Path(downloaded).stat().st_size == 0

    def test_upload_download_file_with_special_characters_local(self, local_experiment, tmp_path):
        """Test upload/download of file with special characters in name."""
        special_file = tmp_path / "file-with_special.chars@123.txt"
        special_file.write_text("Special content")

        with local_experiment(name="special-chars-download", project="test").run as experiment:
            # Upload
            upload_result = experiment.files(
                file_path=str(special_file),
                prefix="/files"
            ).save()

            # Download
            download_path = tmp_path / "downloaded_special.txt"
            downloaded = experiment.files(
                file_id=upload_result["id"],
                dest_path=str(download_path)
            ).download()

        # Verify content
        assert Path(downloaded).read_text() == "Special content"

    @pytest.mark.remote
    def test_upload_download_large_file_remote(self, remote_experiment, sample_files, tmp_path):
        """Test uploading and downloading large file in remote mode."""
        with remote_experiment(name="large-remote-download", project="test").run as experiment:
            # Upload large file
            upload_result = experiment.files(
                file_path=sample_files["large"],
                prefix="/large"
            ).save()

            # Download it back
            download_path = tmp_path / "downloaded_large_remote.bin"
            downloaded = experiment.files(
                file_id=upload_result["id"],
                dest_path=str(download_path)
            ).download()

        # Verify size
        assert Path(downloaded).stat().st_size == 1024 * 100

if __name__ == "__main__":
    """Run all tests with pytest."""
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
