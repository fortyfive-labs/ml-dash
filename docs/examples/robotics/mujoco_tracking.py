"""
MuJoCo Robot Tracking Example

This example demonstrates:
- Rendering frames from MuJoCo simulation
- Saving frames as JPEG images
- Tracking robot end-effector position over time
- Aligning frames with track data using step indices
"""

import mujoco
import numpy as np
from ml_dash import Experiment

# MuJoCo robot XML configuration
xml_content = """
<mujoco model="robot_data_collector">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    <option integrator="RK4" timestep="0.01" gravity="0 0 -9.81"/>

    <asset>
        <texture builtin="checker" height="100" name="texplane" rgb1="0.1 0.1 0.1" rgb2="0.3 0.3 0.3" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" texture="texplane"/>
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 5" dir="0 0 -1"/>
        <geom name="floor" pos="0 0 0" size="2 2 .1" type="plane" material="MatPlane"/>

        <body name="base" pos="0 0 0.05">
            <geom size="0.1 0.1 0.05" type="box" rgba="0.7 0.7 0.7 1"/>
            <body name="link1" pos="0 0 0.1">
                <joint name="joint1" axis="0 0 1" />
                <geom size="0.03 0.2" type="capsule" rgba="0.2 0.5 0.8 1" pos="0 0 0.2"/>
                <body name="link2" pos="0 0 0.4">
                    <joint name="joint2" axis="0 1 0" />
                    <geom size="0.03 0.2" type="capsule" rgba="0.8 0.2 0.2 1" pos="0.2 0 0" quat="0.707 0 0.707 0"/>
                    <site name="end_effector" pos="0.4 0 0" size="0.02" rgba="1 1 0 1"/>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor joint="joint1" ctrlrange="-10 10" gear="10"/>
        <motor joint="joint2" ctrlrange="-10 10" gear="10"/>
    </actuator>
</mujoco>
"""


def main():
    """Run MuJoCo simulation with tracking."""

    with Experiment(
        prefix="examples/robotics/mujoco-tracking",
        tags=["mujoco", "tracking", "example"],
        dash_url='http://localhost:3000',  # Use local server
        # dash_url="https://api.dash.ml",  # Or use remote server
    ).run as experiment:

        experiment.log("Initializing MuJoCo model...")

        try:
            model = mujoco.MjModel.from_xml_string(xml_content)
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=480, width=640)
        except Exception as e:
            experiment.log(f"Model loading failed: {e}")
            return

        steps = 400
        record_interval = 4  # Record every 4 steps

        experiment.log(f"Starting headless data collection...")

        for i in range(steps):
            # Apply time-varying control signals for smooth trajectories
            # Using different frequency sine waves creates complex curves
            ctrl_1 = 4.0 * np.sin(i * 0.03)
            ctrl_2 = 2.5 * np.sin(i * 0.05)
            data.ctrl[0] = ctrl_1
            data.ctrl[1] = ctrl_2

            # Step physics simulation
            mujoco.mj_step(model, data)

            # Record data at intervals
            if i % record_interval == 0:
                # Render current frame
                renderer.update_scene(data)
                pixels = renderer.render()

                # Get end effector world position
                ee_pos = data.site_xpos[model.site('end_effector').id].copy()

                # Save frame with step index
                frame_filename = f"frame_{i:05d}.jpg"
                experiment.files("robot/frames").save_image(
                    pixels,
                    to=frame_filename,
                    quality=85
                )

                # Track position (aligned with frame by step index)
                experiment.track("robot/position").append({
                    "step": i,
                    "frame": frame_filename,  # Reference to frame
                    "x": float(ee_pos[0]),
                    "y": float(ee_pos[1]),
                    "z": float(ee_pos[2])
                })

                # Track control signals
                experiment.track("robot/control").append({
                    "step": i,
                    "ctrl_1": float(ctrl_1),
                    "ctrl_2": float(ctrl_2)
                })

                # Log progress
                experiment.log(
                    f"Frame {i:3d}: x={ee_pos[0]:7.4f}, "
                    f"y={ee_pos[1]:7.4f}, z={ee_pos[2]:7.4f}"
                )

        experiment.log("-" * 50)
        experiment.log(f"Data collection successful!")
        experiment.log(f"Recorded {steps // record_interval} frames")
        experiment.log("-" * 50)


if __name__ == "__main__":
    main()
