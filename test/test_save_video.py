"""Comprehensive tests for save_video() functionality in both local and remote modes."""

import getpass

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def disable_buffering(monkeypatch):
    """Disable buffering for video tests."""
    monkeypatch.setenv("ML_DASH_BUFFER_ENABLED", "false")


class TestVideoBasics:
  """Tests for basic video save operations."""

  def test_save_video_user_example_local(self, local_experiment, tmp_proj):
    """Test exact user example from requirements in local mode."""

    def im(x, y):
      canvas = np.zeros((200, 200))
      for i in range(200):
        for j in range(200):
          if x - 5 < i < x + 5 and y - 5 < j < y + 5:
            canvas[i, j] = 1
      return canvas

    with local_experiment("57block/test/video-test").run as experiment:
      frames = [im(100 + i, 80) for i in range(20)]
      result = experiment.files(prefix="/videos").save_video(
        frames, to="test_video.mp4"
      )

      assert result["filename"] == "test_video.mp4"
      assert result["sizeBytes"] > 0
      assert "checksum" in result

    # Verify file was saved
    files_dir = tmp_proj / getpass.getuser() / "test/video-test/files"
    assert files_dir.exists()
    saved_videos = list(files_dir.glob("**/test_video.mp4"))
    assert len(saved_videos) == 1

  @pytest.mark.remote
  def test_save_video_user_example_remote(self, remote_experiment):
    """Test exact user example from requirements in remote mode."""

    def im(x, y):
      canvas = np.zeros((200, 200))
      for i in range(200):
        for j in range(200):
          if x - 5 < i < x + 5 and y - 5 < j < y + 5:
            canvas[i, j] = 1
      return canvas

    with remote_experiment("57block/test/video-test-remote").run as experiment:
      frames = [im(100 + i, 80) for i in range(20)]
      result = experiment.files(prefix="/videos").save_video(
        frames, to="test_video.mp4"
      )

      assert result["filename"] == "test_video.mp4"
      assert result["sizeBytes"] > 0

  def test_save_video_grayscale_local(self, local_experiment):
    """Test grayscale frame video in local mode."""
    with local_experiment("57block/test/video-grayscale").run as experiment:
      frames = [np.random.rand(100, 100) for _ in range(10)]
      result = experiment.files(prefix="/videos").save_video(frames, to="grayscale.mp4")

      assert result["filename"] == "grayscale.mp4"
      assert result["sizeBytes"] > 0

  @pytest.mark.remote
  def test_save_video_grayscale_remote(self, remote_experiment):
    """Test grayscale frame video in remote mode."""
    with remote_experiment(
      "test-user/test/video-grayscale-remote"
    ).run as experiment:
      frames = [np.random.rand(100, 100) for _ in range(10)]
      result = experiment.files(prefix="/videos").save_video(frames, to="grayscale.mp4")

      assert result["filename"] == "grayscale.mp4"


class TestVideoFormats:
  """Tests for different video formats and frame types."""

  def test_save_video_rgb_local(self, local_experiment):
    """Test RGB frame video."""
    with local_experiment("57block/test/video-rgb").run as experiment:
      frames = [np.random.rand(100, 100, 3) for _ in range(10)]
      result = experiment.files(prefix="/videos").save_video(frames, to="rgb.mp4")

      assert result["filename"] == "rgb.mp4"
      assert result["sizeBytes"] > 0

  def test_save_video_gif_local(self, local_experiment):
    """Test GIF format."""
    with local_experiment("57block/test/video-gif").run as experiment:
      frames = [np.random.rand(50, 50) for _ in range(5)]
      result = experiment.files(prefix="/videos").save_video(frames, to="animation.gif")

      assert result["filename"] == "animation.gif"
      assert result["sizeBytes"] > 0

  @pytest.mark.remote
  def test_save_video_gif_remote(self, remote_experiment):
    """Test GIF format in remote mode."""
    with remote_experiment("57block/test/video-gif-remote").run as experiment:
      frames = [np.random.rand(50, 50) for _ in range(5)]
      result = experiment.files(prefix="/videos").save_video(frames, to="animation.gif")

      assert result["filename"] == "animation.gif"

  def test_save_video_stacked_array_local(self, local_experiment):
    """Test stacked numpy array input."""
    with local_experiment("57block/test/video-stacked").run as experiment:
      frames = np.random.rand(10, 100, 100)
      result = experiment.files(prefix="/videos").save_video(frames, to="stacked.mp4")

      assert result["filename"] == "stacked.mp4"
      assert result["sizeBytes"] > 0


class TestVideoParameters:
  """Tests for video encoding parameters."""

  def test_save_video_custom_fps_local(self, local_experiment):
    """Test custom FPS parameter."""
    with local_experiment("57block/test/video-fps").run as experiment:
      frames = [np.random.rand(100, 100) for _ in range(10)]
      result = experiment.files(prefix="/videos").save_video(
        frames, to="fps30.mp4", fps=30
      )

      assert result["filename"] == "fps30.mp4"
      assert result["sizeBytes"] > 0

  @pytest.mark.remote
  def test_save_video_custom_fps_remote(self, remote_experiment):
    """Test custom FPS in remote mode."""
    with remote_experiment("57block/test/video-fps-remote").run as experiment:
      frames = [np.random.rand(100, 100) for _ in range(10)]
      result = experiment.files(prefix="/videos").save_video(
        frames, to="fps30.mp4", fps=30
      )

      assert result["filename"] == "fps30.mp4"

  def test_save_video_with_kwargs_local(self, local_experiment):
    """Test passing additional imageio kwargs."""
    with local_experiment("57block/test/video-kwargs").run as experiment:
      frames = [np.random.rand(100, 100, 3) for _ in range(10)]
      result = experiment.files(prefix="/videos").save_video(
        frames, to="quality.mp4", fps=30, quality=8
      )

      assert result["filename"] == "quality.mp4"
      assert result["sizeBytes"] > 0


class TestVideoEdgeCases:
  """Tests for edge cases and error handling."""

  def test_save_video_empty_frames_error(self, local_experiment):
    """Test error handling for empty frame list."""
    with local_experiment("57block/test/video-error").run as experiment:
      with pytest.raises(ValueError, match="frame_stack is empty"):
        experiment.files(prefix="/videos").save_video([], to="empty.mp4")

  def test_save_video_float32_frames_local(self, local_experiment):
    """Test float32 frames are converted correctly."""
    with local_experiment("57block/test/video-float32").run as experiment:
      frames = [np.random.rand(100, 100).astype(np.float32) for _ in range(5)]
      result = experiment.files(prefix="/videos").save_video(frames, to="float32.mp4")

      assert result["filename"] == "float32.mp4"
      assert result["sizeBytes"] > 0

  def test_save_video_uint8_frames_local(self, local_experiment):
    """Test uint8 frames are handled correctly."""
    with local_experiment("57block/test/video-uint8").run as experiment:
      frames = [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(5)]
      result = experiment.files(prefix="/videos").save_video(frames, to="uint8.mp4")

      assert result["filename"] == "uint8.mp4"
      assert result["sizeBytes"] > 0


if __name__ == "__main__":
  """Run all tests with pytest."""
  import sys

  sys.exit(pytest.main([__file__, "-v"]))
