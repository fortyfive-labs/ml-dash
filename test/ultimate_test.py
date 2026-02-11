import mujoco
import numpy as np

# Configure RUN BEFORE importing dxp
from ml_dash import RUN
RUN.prefix = "tom_tao_e4c2c9/robot9/examples/tracking-data-collection1"

# Now import dxp - it will use the RUN.prefix we just set
from ml_dash.auto_start import dxp

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
    with dxp.run:
        # Set simulation parameters
        dxp.params.set(
            timestep=0.01,
            integrator="RK4",
            gravity=-9.81,
            control_freq_1=0.03,
            control_freq_2=0.05,
            control_amp_1=4.0,
            control_amp_2=2.5,
            total_steps=400,
            record_interval=4,
            render_height=480,
            render_width=640,
            num_joints=2,
        )

        dxp.log("正在初始化 MuJoCo 模型...")
        try:
            model = mujoco.MjModel.from_xml_string(xml_content)
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=480, width=640)
        except Exception as e:
            dxp.log(f"模型加载失败: {e}")
            return

        # 存储容器
        target_pos = np.array([0.3, 0.2, 0.5])  # 目标位置
        cumulative_reward = 0.0

        steps = 400
        record_interval = 4  # 采样间隔

        dxp.log(f"正在后台采集数据 (Headless 模式)...")

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
                dxp.files("robot/position").save_image(pixels, to=f"frame_{i}.jpg")
                # 获取末端 site 的世界坐标 (Tracks)
                ee_pos = data.site_xpos[model.site('end_effector').id].copy()
                dxp.tracks("robot/position").append(e=ee_pos, _ts=i)

                # 计算训练指标
                # Reward: 负的距离误差 (越接近目标奖励越高)
                distance_to_target = np.linalg.norm(ee_pos - target_pos)
                reward = -distance_to_target
                cumulative_reward += reward

                # Loss: 模拟策略损失 (随训练进度减小)
                policy_loss = 1.0 * np.exp(-i / 200.0) + 0.1 * np.random.random()
                value_loss = 0.5 * np.exp(-i / 150.0) + 0.05 * np.random.random()

                # 记录训练指标
                dxp.metrics("train").log(
                    step=i,
                    reward=reward,
                    cumulative_reward=cumulative_reward,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    distance_to_target=distance_to_target,
                )

                # 实时打印轨迹数据
                dxp.log(f"Frame {i:3d}: x={ee_pos[0]:7.4f}, y={ee_pos[1]:7.4f}, z={ee_pos[2]:7.4f}")

        dxp.log("-" * 40)
        dxp.log(f"数据采集成功！")
        dxp.log("-" * 40)

if __name__ == "__main__":
    main()