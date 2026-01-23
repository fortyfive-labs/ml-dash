"""Test numpy array serialization in tracks."""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from ml_dash import Experiment


def test_numpy_arrays_local():
    """Test that numpy arrays are properly serialized in local mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with Experiment(
            prefix="test-user/test-project/numpy-test",
            dash_root=tmp_dir
        ).run as exp:
            # Test with numpy arrays
            position = np.array([1.0, 2.0, 3.0])
            velocity = np.array([0.1, 0.2, 0.3])

            exp.tracks("robot/position").append(
                pos=position,
                vel=velocity,
                _ts=0.0
            )

            # Test with nested numpy arrays
            exp.tracks("robot/state").append(
                camera={"pos": np.array([0, 0, 1]), "rot": np.array([0, 0, 0, 1])},
                body=np.array([0.5, 0.6, 0.7]),
                _ts=0.0
            )

            # Flush to write
            exp.flush()

            # Read back and verify
            data = exp.tracks("robot/position").read(format="json")

            print(f"✓ Successfully logged {data['count']} entries")

            # Verify data is lists, not numpy arrays
            entry = data['entries'][0]
            assert isinstance(entry['pos'], list), f"Expected list, got {type(entry['pos'])}"
            assert isinstance(entry['vel'], list), f"Expected list, got {type(entry['vel'])}"
            assert entry['pos'] == [1.0, 2.0, 3.0]
            assert entry['vel'] == [0.1, 0.2, 0.3]

            print(f"✓ Data correctly serialized as lists")
            print(f"  pos: {entry['pos']}")
            print(f"  vel: {entry['vel']}")

            # Check nested arrays
            state_data = exp.tracks("robot/state").read(format="json")
            state_entry = state_data['entries'][0]
            assert isinstance(state_entry['camera.pos'], list)
            assert isinstance(state_entry['camera.rot'], list)
            assert isinstance(state_entry['body'], list)

            print(f"✓ Nested arrays correctly serialized")
            print(f"  camera.pos: {state_entry['camera.pos']}")
            print(f"  camera.rot: {state_entry['camera.rot']}")
            print(f"  body: {state_entry['body']}")


def test_numpy_scalars():
    """Test that numpy scalar types are properly serialized."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with Experiment(
            prefix="test-user/test-project/numpy-scalar",
            dash_root=tmp_dir
        ).run as exp:
            # Test with numpy scalars
            x = np.float64(1.5)
            y = np.int32(42)
            z = np.bool_(True)

            exp.tracks("robot/data").append(
                x=x,
                y=y,
                z=z,
                _ts=0.0
            )

            exp.flush()

            # Read back
            data = exp.tracks("robot/data").read(format="json")
            entry = data['entries'][0]

            print(f"✓ Numpy scalars serialized")
            print(f"  x={entry['x']} (type: {type(entry['x']).__name__})")
            print(f"  y={entry['y']} (type: {type(entry['y']).__name__})")
            print(f"  z={entry['z']} (type: {type(entry['z']).__name__})")

            assert isinstance(entry['x'], float)
            assert isinstance(entry['y'], int)
            assert isinstance(entry['z'], bool)


if __name__ == "__main__":
    print("Testing numpy array serialization in tracks...\n")

    print("Test 1: Numpy arrays")
    print("-" * 50)
    test_numpy_arrays_local()
    print()

    print("Test 2: Numpy scalars")
    print("-" * 50)
    test_numpy_scalars()
    print()

    print("=" * 50)
    print("✓ All numpy serialization tests passed!")
