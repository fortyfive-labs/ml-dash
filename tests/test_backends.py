"""Tests for storage backends."""

import pytest
import tempfile
import shutil
from pathlib import Path

from ml_dash.backends.local_backend import LocalBackend


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def backend(temp_dir):
    """Create a LocalBackend instance."""
    return LocalBackend(temp_dir)


class TestLocalBackendBasics:
    """Test basic LocalBackend functionality."""

    def test_backend_creation(self, backend, temp_dir):
        """Test creating a backend."""
        assert backend.root_dir == Path(temp_dir).resolve()
        assert backend.root_dir.exists()

    def test_exists_false_for_new_file(self, backend):
        """Test exists returns False for non-existent files."""
        assert not backend.exists("test.txt")

    def test_exists_true_after_write(self, backend):
        """Test exists returns True after writing."""
        backend.write_text("test.txt", "content")
        assert backend.exists("test.txt")


class TestLocalBackendText:
    """Test text operations."""

    def test_write_and_read_text(self, backend):
        """Test writing and reading text."""
        backend.write_text("test.txt", "Hello, world!")
        content = backend.read_text("test.txt")
        assert content == "Hello, world!"

    def test_append_text(self, backend):
        """Test appending text."""
        backend.write_text("test.txt", "Line 1\n")
        backend.append_text("test.txt", "Line 2\n")
        backend.append_text("test.txt", "Line 3\n")

        content = backend.read_text("test.txt")
        assert content == "Line 1\nLine 2\nLine 3\n"

    def test_append_to_nonexistent_file(self, backend):
        """Test appending to a file that doesn't exist."""
        backend.append_text("test.txt", "First line\n")
        content = backend.read_text("test.txt")
        assert content == "First line\n"


class TestLocalBackendBinary:
    """Test binary operations."""

    def test_write_and_read_bytes(self, backend):
        """Test writing and reading binary data."""
        data = b"\x00\x01\x02\x03\x04"
        backend.write_bytes("test.bin", data)
        loaded = backend.read_bytes("test.bin")
        assert loaded == data

    def test_write_bytes_creates_directories(self, backend):
        """Test that write_bytes creates parent directories."""
        backend.write_bytes("a/b/c/test.bin", b"data")
        assert backend.exists("a/b/c/test.bin")


class TestLocalBackendDirectories:
    """Test directory operations."""

    def test_makedirs(self, backend):
        """Test creating directories."""
        backend.makedirs("a/b/c")
        path = backend._resolve_path("a/b/c")
        assert path.exists()
        assert path.is_dir()

    def test_makedirs_exist_ok(self, backend):
        """Test makedirs with exist_ok."""
        backend.makedirs("test")
        backend.makedirs("test", exist_ok=True)  # Should not raise

    def test_list_dir(self, backend):
        """Test listing directory contents."""
        backend.write_text("file1.txt", "content")
        backend.write_text("file2.txt", "content")
        backend.makedirs("subdir")

        files = backend.list_dir()
        assert "file1.txt" in files
        assert "file2.txt" in files
        assert "subdir" in files

    def test_list_dir_subdir(self, backend):
        """Test listing subdirectory."""
        backend.write_text("dir/file1.txt", "content")
        backend.write_text("dir/file2.txt", "content")

        files = backend.list_dir("dir")
        assert "file1.txt" in files
        assert "file2.txt" in files

    def test_list_dir_nonexistent(self, backend):
        """Test listing non-existent directory."""
        files = backend.list_dir("nonexistent")
        assert files == []


class TestLocalBackendDelete:
    """Test delete operations."""

    def test_delete_file(self, backend):
        """Test deleting a file."""
        backend.write_text("test.txt", "content")
        assert backend.exists("test.txt")

        backend.delete("test.txt")
        assert not backend.exists("test.txt")

    def test_delete_nonexistent(self, backend):
        """Test deleting non-existent file."""
        # Should not raise
        backend.delete("nonexistent.txt")


class TestLocalBackendURL:
    """Test URL generation."""

    def test_get_url_existing_file(self, backend):
        """Test getting URL for existing file."""
        backend.write_text("test.txt", "content")
        url = backend.get_url("test.txt")
        assert url.startswith("file://")
        assert "test.txt" in url

    def test_get_url_nonexistent_file(self, backend):
        """Test getting URL for non-existent file."""
        url = backend.get_url("nonexistent.txt")
        assert url is None


class TestLocalBackendPaths:
    """Test path resolution."""

    def test_nested_paths(self, backend):
        """Test nested path handling."""
        backend.write_text("a/b/c/test.txt", "content")
        assert backend.exists("a/b/c/test.txt")

        content = backend.read_text("a/b/c/test.txt")
        assert content == "content"

    def test_path_with_dots(self, backend):
        """Test paths with dots in names."""
        backend.write_text("file.test.txt", "content")
        assert backend.exists("file.test.txt")


class TestLocalBackendIsolation:
    """Test backend isolation."""

    def test_backends_are_isolated(self, temp_dir):
        """Test that different backends are isolated."""
        backend1 = LocalBackend(temp_dir + "/backend1")
        backend2 = LocalBackend(temp_dir + "/backend2")

        backend1.write_text("test.txt", "backend1")
        backend2.write_text("test.txt", "backend2")

        assert backend1.read_text("test.txt") == "backend1"
        assert backend2.read_text("test.txt") == "backend2"
