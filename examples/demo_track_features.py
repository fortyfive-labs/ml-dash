"""
Demo: Track Features - Timestamp Inheritance and Slicing

This example demonstrates:
1. Timestamp inheritance with _ts=-1 for synchronized multi-modal data
2. Track slicing with iteration
3. Floor-match timestamp queries with findByTime()
4. Batch queries with list of timestamps
"""

from ml_dash import Experiment
import time


def demo_timestamp_inheritance():
    """Demonstrate timestamp inheritance for synchronized data."""
    print("=" * 60)
    print("Demo 1: Timestamp Inheritance (_ts=-1)")
    print("=" * 60)

    with Experiment(prefix="demo/timestamp-inheritance", dash_root=".dash").run as experiment:
        # Simulate robot data collection with synchronized sensors
        for step in range(10):
            # First track auto-generates timestamp
            experiment.tracks("robot/pose").append(
                step=step,
                position=[1.0 + step * 0.1, 2.0, 3.0],
                orientation=[0.0, 0.0, step * 0.01]
            )

            # Other tracks inherit the same timestamp using _ts=-1
            experiment.tracks("camera/left").append(
                step=step,
                width=640,
                height=480,
                _ts=-1  # Inherits timestamp from robot/pose
            )

            experiment.tracks("camera/right").append(
                step=step,
                width=640,
                height=480,
                _ts=-1  # Same timestamp
            )

            experiment.tracks("robot/velocity").append(
                step=step,
                linear=[0.1, 0.0, 0.0],
                angular=[0.0, 0.0, 0.05],
                _ts=-1  # Same timestamp
            )

            time.sleep(0.01)  # Simulate data collection delay

        experiment.tracks.flush()

        # Verify timestamps are synchronized
        pose_data = experiment.tracks("robot/pose").read(format="json")
        camera_data = experiment.tracks("camera/left").read(format="json")

        print(f"\nCollected {len(pose_data['entries'])} synchronized timesteps")
        print("\nFirst 3 entries verification:")
        for i in range(3):
            pose_ts = pose_data["entries"][i]["timestamp"]
            camera_ts = camera_data["entries"][i]["timestamp"]
            print(f"  Step {i}: pose_ts={pose_ts:.6f}, camera_ts={camera_ts:.6f}, "
                  f"match={pose_ts == camera_ts}")


def demo_track_slicing():
    """Demonstrate track slicing with iteration."""
    print("\n" + "=" * 60)
    print("Demo 2: Track Slicing and Iteration")
    print("=" * 60)

    with Experiment(prefix="demo/track-slicing", dash_root=".dash").run as experiment:
        # Create track with known timestamps
        for i in range(20):
            experiment.tracks("sensor/data").append(
                step=i,
                value=i * 10,
                _ts=float(i)
            )

        experiment.tracks.flush()

        # Create slice for analysis window [5.0, 15.0]
        track_slice = experiment.tracks("sensor/data").slice(
            start_timestamp=5.0,
            end_timestamp=15.0
        )

        print(f"\nSlice contains {len(track_slice)} entries")
        print("\nIterating through slice:")
        for entry in track_slice:
            print(f"  t={entry['timestamp']:4.1f}, step={entry['step']:2d}, "
                  f"value={entry['value']:3d}")


def demo_floor_match_queries():
    """Demonstrate floor-match timestamp queries."""
    print("\n" + "=" * 60)
    print("Demo 3: Floor-Match Timestamp Queries")
    print("=" * 60)

    with Experiment(prefix="demo/floor-match", dash_root=".dash").run as experiment:
        # Create track with sparse timestamps [1.0, 3.0, 5.0, 7.0, 9.0]
        for i in [1, 3, 5, 7, 9]:
            experiment.tracks("sparse/sensor").append(
                value=i * 100,
                _ts=float(i)
            )

        experiment.tracks.flush()

        track_slice = experiment.tracks("sparse/sensor").slice()

        # Single timestamp queries with floor matching
        print("\nSingle timestamp queries (floor match):")
        query_times = [1.0, 2.5, 3.0, 4.5, 5.5, 8.0, 10.0]

        for query_ts in query_times:
            entry = track_slice.findByTime(query_ts)
            print(f"  query={query_ts:4.1f} -> matched_ts={entry['timestamp']:4.1f}, "
                  f"value={entry['value']:3d}")


def demo_batch_queries():
    """Demonstrate batch queries with list of timestamps."""
    print("\n" + "=" * 60)
    print("Demo 4: Batch Queries with List of Timestamps")
    print("=" * 60)

    with Experiment(prefix="demo/batch-queries", dash_root=".dash").run as experiment:
        # Create track with data every second
        for i in range(20):
            experiment.tracks("continuous/sensor").append(
                value=i * 5,
                _ts=float(i)
            )

        experiment.tracks.flush()

        track_slice = experiment.tracks("continuous/sensor").slice()

        # Batch query multiple timestamps at once
        query_times = [2.0, 5.5, 10.0, 15.7, 19.0]
        entries = track_slice.findByTime(query_times)

        print(f"\nBatch query for {len(query_times)} timestamps:")
        for i, (query_ts, entry) in enumerate(zip(query_times, entries)):
            print(f"  [{i}] query={query_ts:5.1f} -> matched_ts={entry['timestamp']:4.1f}, "
                  f"value={entry['value']:3d}")


def demo_synchronized_analysis():
    """Demonstrate analyzing synchronized multi-modal data."""
    print("\n" + "=" * 60)
    print("Demo 5: Synchronized Multi-Modal Analysis")
    print("=" * 60)

    with Experiment(prefix="demo/sync-analysis", dash_root=".dash").run as experiment:
        # Log synchronized robot data with explicit timestamps
        for step in range(15):
            # All tracks share same timestamp via _ts=-1
            experiment.tracks("robot/pose").append(
                step=step,
                x=step * 0.1,
                y=step * 0.05,
                z=1.0,
                _ts=float(step)  # Use step as timestamp for easy querying
            )

            experiment.tracks("robot/velocity").append(
                step=step,
                vx=0.1,
                vy=0.05,
                _ts=-1
            )

            experiment.tracks("sensor/lidar").append(
                step=step,
                distance=5.0 - step * 0.1,
                _ts=-1
            )

        experiment.tracks.flush()

        # Analyze synchronized data at specific times
        pose_slice = experiment.tracks("robot/pose").slice()
        velocity_slice = experiment.tracks("robot/velocity").slice()
        lidar_slice = experiment.tracks("sensor/lidar").slice()

        # Query all tracks at specific analysis points
        analysis_times = [3.0, 7.5, 12.0]
        print(f"\nAnalyzing {len(analysis_times)} time points:")

        for t in analysis_times:
            pose = pose_slice.findByTime(t)
            vel = velocity_slice.findByTime(t)
            lidar = lidar_slice.findByTime(t)

            # All entries have same timestamp (synchronized)
            assert pose["timestamp"] == vel["timestamp"] == lidar["timestamp"]

            print(f"\n  Time t={t:.1f} (matched: {pose['timestamp']:.1f}):")
            print(f"    Position: ({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f})")
            print(f"    Velocity: ({vel['vx']:.2f}, {vel['vy']:.2f})")
            print(f"    LIDAR distance: {lidar['distance']:.2f}m")


if __name__ == "__main__":
    demo_timestamp_inheritance()
    demo_track_slicing()
    demo_floor_match_queries()
    demo_batch_queries()
    demo_synchronized_analysis()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
