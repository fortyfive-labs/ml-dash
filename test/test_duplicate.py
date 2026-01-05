"""Tests for FileBuilder.duplicate() method."""
import pytest
import tempfile
from pathlib import Path


class TestDuplicateLocal:
    """Test duplicate() in local mode."""

    def test_duplicate_with_file_id(self, local_experiment):
        """Test duplicating a file using file ID."""
        exp = local_experiment(name="test_duplicate_id", project="test")
        exp.run.start()

        # Create a test file and upload it
        test_content = b"test checkpoint data"
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            # Upload original file
            original = exp.files("checkpoints").upload(temp_path)

            # Duplicate to new location
            duplicate = exp.files().duplicate(original['id'], to="models/latest.pt")

            # Verify duplicate has correct metadata
            assert duplicate['filename'] == 'latest.pt'
            assert duplicate['path'] == '/models'
            assert duplicate['checksum'] == original['checksum']
            assert duplicate['sizeBytes'] == original['sizeBytes']
            assert duplicate['id'] != original['id']  # Different file ID

            # Verify both files exist
            files_list = exp.files().list()
            filenames = [f['filename'] for f in files_list]
            assert original['filename'] in filenames
            assert 'latest.pt' in filenames

        finally:
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            exp.run.complete()

    def test_duplicate_with_metadata_dict(self, local_experiment):
        """Test duplicating a file using metadata dict from save()."""
        exp = local_experiment(name="test_duplicate_dict", project="test")
        exp.run.start()

        # Create a test file
        test_content = b"checkpoint v100"
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            # Upload and get metadata
            snapshot = exp.files("checkpoints").upload(temp_path)

            # Duplicate using metadata dict
            latest = exp.files().duplicate(snapshot, to="checkpoints/best.pt")

            # Verify
            assert latest['filename'] == 'best.pt'
            assert latest['path'] == '/checkpoints'
            assert latest['checksum'] == snapshot['checksum']

        finally:
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            exp.run.complete()

    def test_duplicate_various_path_formats(self, local_experiment):
        """Test duplicate() with various path formats."""
        exp = local_experiment(name="test_duplicate_paths", project="test")
        exp.run.start()

        # Create test file
        test_content = b"test data"
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            original = exp.files().upload(temp_path)

            # Test different path formats
            test_cases = [
                ("model.pt", "/", "model.pt"),
                ("models/latest.pt", "/models", "latest.pt"),
                ("/checkpoints/best.pt", "/checkpoints", "best.pt"),
            ]

            for to_path, expected_prefix, expected_filename in test_cases:
                dup = exp.files().duplicate(original, to=to_path)
                assert dup['path'] == expected_prefix, f"Failed for path: {to_path}"
                assert dup['filename'] == expected_filename, f"Failed for path: {to_path}"

        finally:
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            exp.run.complete()

    def test_duplicate_source_not_found(self, local_experiment):
        """Test that duplicate() raises error for non-existent source."""
        exp = local_experiment(name="test_duplicate_error", project="test")
        exp.run.start()

        with pytest.raises(Exception):  # Should raise an error when file not found
            exp.files().duplicate("nonexistent-file-id", to="output.pt")

        exp.run.complete()

    def test_duplicate_invalid_source(self, local_experiment):
        """Test that duplicate() raises ValueError for invalid source."""
        exp = local_experiment(name="test_duplicate_invalid", project="test")
        exp.run.start()

        # Test with invalid types
        with pytest.raises(ValueError, match="source must be"):
            exp.files().duplicate(123, to="output.pt")

        with pytest.raises(ValueError, match="source must be"):
            exp.files().duplicate({"wrong": "key"}, to="output.pt")

        with pytest.raises(ValueError, match="source must be"):
            exp.files().duplicate(None, to="output.pt")

        exp.run.complete()

    def test_duplicate_overwrites_existing(self, local_experiment):
        """Test that duplicate() overwrites existing file at target location."""
        exp = local_experiment(name="test_duplicate_overwrite", project="test")
        exp.run.start()

        # Create two different files
        content1 = b"version 1"
        content2 = b"version 2 updated"

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt', delete=False) as f:
            f.write(content1)
            temp_path1 = f.name

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt', delete=False) as f:
            f.write(content2)
            temp_path2 = f.name

        try:
            # Upload first version and duplicate to "latest"
            v1 = exp.files().upload(temp_path1)
            latest1 = exp.files().duplicate(v1, to="models/latest.pt")
            checksum1 = latest1['checksum']

            # Upload second version and duplicate to same "latest" location
            v2 = exp.files().upload(temp_path2)
            latest2 = exp.files().duplicate(v2, to="models/latest.pt")
            checksum2 = latest2['checksum']

            # Verify the latest has been updated
            assert checksum2 != checksum1
            assert checksum2 == v2['checksum']

            # Verify only one "latest.pt" exists
            files_list = exp.files(prefix="/models").list()
            latest_files = [f for f in files_list if f['filename'] == 'latest.pt']
            assert len(latest_files) == 1
            assert latest_files[0]['checksum'] == checksum2

        finally:
            import os
            for path in [temp_path1, temp_path2]:
                if os.path.exists(path):
                    os.unlink(path)
            exp.run.complete()


@pytest.mark.remote
class TestDuplicateRemote:
    """Test duplicate() in remote mode."""

    def test_duplicate_remote_basic(self, remote_experiment):
        """Test basic duplicate in remote mode."""
        exp = remote_experiment(name="test_duplicate_remote", project="test_files")
        exp.run.start()

        # Create test file
        test_content = b"remote checkpoint"
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            # Upload and duplicate
            original = exp.files("models").upload(temp_path)
            duplicate = exp.files().duplicate(original, to="models/latest.pt")

            # Verify
            assert duplicate['filename'] == 'latest.pt'
            assert duplicate['checksum'] == original['checksum']

        finally:
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            exp.run.complete()

if __name__ == "__main__":
    """Run all tests with pytest."""
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
