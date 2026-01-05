"""Tests for the new fluent file operations API."""
import json
import pytest
import hashlib
from pathlib import Path


class TestFluentFileSave:
    """Tests for the new fluent save() API."""

    def test_save_existing_file_path(self, local_experiment, sample_files):
        """Test saving an existing file using path argument."""
        with local_experiment(name="fluent-save-file", project="test").run as exp:
            result = exp.files("models").save(sample_files["model"])

            assert result["filename"] == "model.txt"
            assert result["path"] == "/models"
            assert result["sizeBytes"] > 0
            assert "checksum" in result

    def test_save_with_to_parameter(self, local_experiment, tmp_path):
        """Test saving bytes with to parameter."""
        with local_experiment(name="fluent-save-to", project="test").run as exp:
            result = exp.files("data").save(b"hello world", to="greeting.bin")

            assert result["filename"] == "greeting.bin"
            assert result["path"] == "/data"

    def test_save_dict_as_json(self, local_experiment, tmp_path):
        """Test saving dict as JSON."""
        with local_experiment(name="fluent-save-json", project="test").run as exp:
            config = {"model": "resnet50", "lr": 0.001}
            result = exp.files("configs").save(config, to="config.json")

            assert result["filename"] == "config.json"
            assert result["path"] == "/configs"

            # Verify we can download and read it
            downloaded = exp.files(file_id=result["id"]).download()
            content = json.loads(Path(downloaded).read_text())
            assert content == config

    def test_save_list_as_json(self, local_experiment):
        """Test saving list as JSON."""
        with local_experiment(name="fluent-save-list", project="test").run as exp:
            data = [1, 2, 3, {"key": "value"}]
            result = exp.files("data").save(data, to="array.json")

            assert result["filename"] == "array.json"

    def test_files_save_direct(self, local_experiment, sample_files):
        """Test experiment.folder.save() direct method."""
        with local_experiment(name="fluent-direct-save", project="test").run as exp:
            result = exp.folder.save(sample_files["model"])

            assert result["filename"] == "model.txt"

    def test_files_save_with_path(self, local_experiment):
        """Test experiment.folder.save() with to parameter including path."""
        with local_experiment(name="fluent-direct-save-path", project="test").run as exp:
            result = exp.folder.save({"key": "value"}, to="configs/settings.json")

            assert result["filename"] == "settings.json"
            assert result["path"] == "/configs"


class TestFluentSaveText:
    """Tests for save_text() method."""

    def test_save_text_basic(self, local_experiment, tmp_path):
        """Test basic save_text usage."""
        with local_experiment(name="save-text", project="test").run as exp:
            content = "Hello, World!\nThis is a test."
            result = exp.files("texts").save_text(content, to="greeting.txt")

            assert result["filename"] == "greeting.txt"
            assert result["path"] == "/texts"

            # Download and verify
            downloaded = exp.files(file_id=result["id"]).download()
            assert Path(downloaded).read_text() == content

    def test_save_text_direct(self, local_experiment):
        """Test experiment.folder.save_text() direct method."""
        with local_experiment(name="save-text-direct", project="test").run as exp:
            result = exp.folder.save_text("content", to="file.txt")

            assert result["filename"] == "file.txt"

    def test_save_text_with_path(self, local_experiment):
        """Test save_text with path including prefix."""
        with local_experiment(name="save-text-path", project="test").run as exp:
            result = exp.folder.save_text("yaml: content", to="configs/view.yaml")

            assert result["filename"] == "view.yaml"
            assert result["path"] == "/configs"


class TestFluentSaveJson:
    """Tests for save_json() method."""

    def test_save_json_dict(self, local_experiment, tmp_path):
        """Test save_json with dict."""
        with local_experiment(name="save-json-dict", project="test").run as exp:
            data = {"hey": "yo", "count": 42}
            result = exp.files("configs").save_json(data, to="config.json")

            assert result["filename"] == "config.json"

            # Download and verify
            downloaded = exp.files(file_id=result["id"]).download()
            content = json.loads(Path(downloaded).read_text())
            assert content == data

    def test_save_json_direct(self, local_experiment):
        """Test experiment.folder.save_json() direct method."""
        with local_experiment(name="save-json-direct", project="test").run as exp:
            result = exp.folder.save_json({"key": "value"}, to="data.json")

            assert result["filename"] == "data.json"


class TestFluentSaveBlob:
    """Tests for save_blob() method."""

    def test_save_blob_basic(self, local_experiment, tmp_path):
        """Test basic save_blob usage."""
        with local_experiment(name="save-blob", project="test").run as exp:
            data = b"\x00\x01\x02\x03\x04\x05"
            result = exp.files("data").save_blob(data, to="binary.bin")

            assert result["filename"] == "binary.bin"
            assert result["sizeBytes"] == 6

            # Download and verify
            downloaded = exp.files(file_id=result["id"]).download()
            assert Path(downloaded).read_bytes() == data

    def test_save_blob_direct(self, local_experiment):
        """Test experiment.folder.save_blob() direct method."""
        with local_experiment(name="save-blob-direct", project="test").run as exp:
            result = exp.folder.save_blob(b"data", to="file.bin")

            assert result["filename"] == "file.bin"


class TestFluentList:
    """Tests for fluent list() API."""

    def test_list_all(self, local_experiment, sample_files):
        """Test listing all files."""
        with local_experiment(name="list-all", project="test").run as exp:
            exp.files("models").save(sample_files["model"])
            exp.files("configs").save(sample_files["config"])

            files = exp.files().list()

            assert len(files) == 2
            filenames = [f["filename"] for f in files]
            assert "model.txt" in filenames
            assert "config.json" in filenames

    def test_list_by_prefix(self, local_experiment, sample_files):
        """Test listing files by prefix."""
        with local_experiment(name="list-prefix", project="test").run as exp:
            exp.files("models").save(sample_files["model"])
            exp.files("configs").save(sample_files["config"])
            exp.files("models").save(sample_files["results"])

            # List only models
            files = exp.files("models").list()

            assert len(files) == 2

    def test_list_with_glob_pattern(self, local_experiment, sample_files):
        """Test listing with glob pattern."""
        with local_experiment(name="list-glob", project="test").run as exp:
            exp.files("data").save(sample_files["model"])  # .txt
            exp.files("data").save(sample_files["config"])  # .json
            exp.files("data").save(sample_files["results"])  # .csv

            # List only JSON files
            json_files = exp.files("data").list("*.json")
            assert len(json_files) == 1
            assert json_files[0]["filename"] == "config.json"

            # List txt files
            txt_files = exp.files("data").list("*.txt")
            assert len(txt_files) == 1

    def test_list_direct_with_pattern(self, local_experiment, sample_files):
        """Test experiment.folder.list() with pattern."""
        with local_experiment(name="list-direct", project="test").run as exp:
            exp.files("models").save(sample_files["model"])
            exp.files("configs").save(sample_files["config"])

            files = exp.folder.list("*.txt")
            assert len(files) == 1


class TestFluentDownload:
    """Tests for fluent download() API."""

    def test_download_by_path(self, local_experiment, sample_files, tmp_path):
        """Test downloading file by path."""
        with local_experiment(name="download-path", project="test").run as exp:
            exp.files("models").save(sample_files["model"])

            # Download by filename
            downloaded = exp.files("model.txt").download(to=str(tmp_path / "out.txt"))

            assert Path(downloaded).exists()
            assert Path(downloaded).read_text() == Path(sample_files["model"]).read_text()

    def test_download_with_pattern(self, local_experiment, sample_files, tmp_path):
        """Test downloading multiple files with pattern."""
        with local_experiment(name="download-pattern", project="test").run as exp:
            exp.files("data").save(sample_files["model"])  # .txt
            exp.files("data").save(sample_files["config"])  # .json

            # Download all files matching pattern
            dest_dir = tmp_path / "downloads"
            downloaded = exp.files("data").download("*.txt", to=str(dest_dir))

            assert isinstance(downloaded, list)
            assert len(downloaded) == 1
            assert Path(downloaded[0]).exists()

    def test_download_direct_single(self, local_experiment, sample_files, tmp_path):
        """Test experiment.folder.download() for single file."""
        with local_experiment(name="download-direct", project="test").run as exp:
            exp.files("models").save(sample_files["model"])

            downloaded = exp.folder.download("model.txt", to=str(tmp_path / "out.txt"))

            assert Path(downloaded).exists()

    def test_download_direct_with_pattern(self, local_experiment, sample_files, tmp_path):
        """Test experiment.folder.download() with glob pattern."""
        with local_experiment(name="download-direct-glob", project="test").run as exp:
            exp.files("images").save(sample_files["model"])
            exp.files("images").save(sample_files["config"])

            # Download using path/pattern syntax
            dest_dir = tmp_path / "local_images"
            downloaded = exp.folder.download("images/*.txt", to=str(dest_dir))

            assert isinstance(downloaded, list)
            assert len(downloaded) == 1


class TestFluentDelete:
    """Tests for fluent delete() API."""

    def test_delete_by_path(self, local_experiment, sample_files):
        """Test deleting file by path."""
        with local_experiment(name="delete-path", project="test").run as exp:
            exp.files("models").save(sample_files["model"])

            files_before = exp.files().list()
            assert len(files_before) == 1

            # Delete by filename
            result = exp.files("model.txt").delete()

            assert "deletedAt" in result or "id" in result

    def test_delete_with_pattern(self, local_experiment, sample_files):
        """Test deleting multiple files with pattern."""
        with local_experiment(name="delete-pattern", project="test").run as exp:
            exp.files("data").save(sample_files["model"])  # .txt
            exp.files("data").save(sample_files["config"])  # .json
            exp.files("data").save(sample_files["results"])  # .csv

            # Delete all txt files
            results = exp.files("data").delete("*.txt")

            assert isinstance(results, list)
            assert len(results) == 1

    def test_delete_direct(self, local_experiment, sample_files):
        """Test experiment.folder.delete() direct method."""
        with local_experiment(name="delete-direct", project="test").run as exp:
            exp.files("models").save(sample_files["model"])

            result = exp.folder.delete("model.txt")

            assert "deletedAt" in result or "id" in result

    def test_delete_with_path_pattern(self, local_experiment, sample_files):
        """Test experiment.folder.delete() with path/pattern."""
        with local_experiment(name="delete-direct-glob", project="test").run as exp:
            exp.files("images").save(sample_files["model"])
            exp.files("images").save(sample_files["config"])

            results = exp.folder.delete("images/*.txt")

            assert isinstance(results, list)


class TestBackwardsCompatibility:
    """Tests to ensure backwards compatibility with old API."""

    def test_old_api_file_path_prefix(self, local_experiment, sample_files):
        """Test old API with file_path and prefix kwargs still works."""
        with local_experiment(name="compat-upload", project="test").run as exp:
            result = exp.files(
                file_path=sample_files["model"],
                prefix="/models",
                description="Model weights",
                tags=["model"]
            ).save()

            assert result["filename"] == "model.txt"
            assert result["path"] == "/models"

    def test_old_api_download_by_file_id(self, local_experiment, sample_files, tmp_path):
        """Test old API download with file_id still works."""
        with local_experiment(name="compat-download", project="test").run as exp:
            upload_result = exp.files(
                file_path=sample_files["model"],
                prefix="/models"
            ).save()

            file_id = upload_result["id"]

            # Old API download
            downloaded = exp.files(
                file_id=file_id,
                dest_path=str(tmp_path / "model.txt")
            ).download()

            assert Path(downloaded).exists()

    def test_old_api_delete_by_file_id(self, local_experiment, sample_files):
        """Test old API delete with file_id still works."""
        with local_experiment(name="compat-delete", project="test").run as exp:
            upload_result = exp.files(
                file_path=sample_files["model"],
                prefix="/models"
            ).save()

            file_id = upload_result["id"]

            # Old API delete
            result = exp.files(file_id=file_id).delete()

            assert "deletedAt" in result or "id" in result

    def test_old_api_list_with_prefix(self, local_experiment, sample_files):
        """Test old API list with prefix still works."""
        with local_experiment(name="compat-list", project="test").run as exp:
            exp.files(file_path=sample_files["model"], prefix="/models").save()
            exp.files(file_path=sample_files["config"], prefix="/configs").save()

            # Old API list with prefix
            files = exp.files(prefix="/models").list()

            assert len(files) == 1
            assert files[0]["filename"] == "model.txt"


class TestBindrs:
    """Tests for bindrs (file collections)."""

    def test_bindrs_list_placeholder(self, local_experiment, sample_files):
        """Test bindrs.list() placeholder functionality."""
        with local_experiment(name="bindrs-test", project="test").run as exp:
            # Upload files - bindrs filtering will return empty for now
            # as bindrs metadata isn't set on files
            exp.files("models").save(sample_files["model"])

            # This should work but return empty (placeholder)
            files = exp.bindrs("my-bindr").list()

            # Placeholder returns empty list (no files have this bindr)
            assert isinstance(files, list)


if __name__ == "__main__":
    """Run all tests with pytest."""
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
