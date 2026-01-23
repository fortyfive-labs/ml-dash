import mujoco
import numpy as np
import os
from ml_dash import Experiment

# 1. 严格校验过的 XML 资产
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
    # 2. 初始化 MuJoCo
    with Experiment(
        prefix=f"tom_tao_34833x/robot11/examples/tracking-data-collection",
        tags=["sweep", "best"],
        dash_url='http://localhost:3000',  # Use for local server testing
        # dash_url="https://api.dash.ml",  # Use for remote mode
        # dash_root=".dash",  # Local storage directory
    ).run as experiment:
        experiment.log("正在初始化 MuJoCo 模型...")
        try:
            model = mujoco.MjModel.from_xml_string(xml_content)
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=480, width=640)
        except Exception as e:
            experiment.log(f"模型加载失败: {e}")
            return

        # 存储容器

        steps = 400
        record_interval = 4  # 采样间隔

        experiment.log(f"正在后台采集数据 (Headless 模式)...")

        for i in range(steps):
            # 3. 施加时变控制信号，形成平滑轨迹
            # 使用不同频率的正弦波让末端划出复杂的曲线
            ctrl_1 = 4.0 * np.sin(i * 0.03)
            ctrl_2 = 2.5 * np.sin(i * 0.05)
            data.ctrl[0] = ctrl_1
            data.ctrl[1] = ctrl_2

            # 4. 物理仿真推进
            mujoco.mj_step(model, data)

            # 5. 定期采集数据
            if i % record_interval == 0:
                # 渲染当前帧
                renderer.update_scene(data)
                pixels = renderer.render()
                frame_filename = f"frame_{i:05d}.jpg"
         
                # 获取末端 site 的世界坐标 (Tracks)
                ee_pos = data.site_xpos[model.site('end_effector').id].copy()
                experiment.files("robot/camera").save_image(pixels, to=frame_filename)
                experiment.tracks("robot/position").append(e=ee_pos, _ts=i)
                experiment.tracks("robot/camera").append(frame=frame_filename, _ts=i) 
                # 记录执行的动作
                # actions.append([ctrl_1, ctrl_2])

                # 实时打印轨迹数据
                experiment.log(f"Frame {i:3d}: x={ee_pos[0]:7.4f}, y={ee_pos[1]:7.4f}, z={ee_pos[2]:7.4f}")

        experiment.log("-" * 40)
        experiment.log(f"数据采集成功！")
        # experiment.log(f"文件路径: {os.path.abspath(output_file)}")
        # experiment.log(f"采集总帧数: {len(images)}")
        # experiment.log(f"数据结构: Images {np.array(images).shape}, Tracks {np.array(tracks).shape}")
        experiment.log("-" * 40)

if __name__ == "__main__":
    main()