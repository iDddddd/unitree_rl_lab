# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)

概述

本项目为宇树科技（Unitree）机器人提供了一套基于https://github.com/isaac-sim/IsaacLab的强化学习环境。

目前支持宇树科技Go2、H1和G1-29dof机器人。

<div align="center">

<div align="center"> Isaac Lab 仿真 </div> <div align="center">  Mujoco 仿真 </div> <div align="center"> 实体机器人 </div>

g1_sim.gif g1_mujoco.gif g1_real.gif

</div>

安装

• 按照https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html安装Isaac Lab

• 安装Unitree RL IsaacLab独立环境

  • 在Isaac Lab安装目录之外单独克隆此仓库（即不在IsaacLab目录内）：
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    
  • 使用已安装Isaac Lab的Python解释器，以可编辑模式安装库：
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # 重启shell以激活环境变更
    

• 下载宇树机器人描述文件

  方法一：使用USD文件
  • 从https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main下载宇树USD文件，保持文件夹结构
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    
  • 在source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py中配置UNITREE_MODEL_DIR
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    

  方法二：使用URDF文件【推荐】 仅适用于Isaacsim >= 5.0
  • 从https://github.com/unitreerobotics/unitree_ros下载宇树机器人URDF文件

      git clone https://github.com/unitreerobotics/unitree_ros.git
      
  • 在source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py中配置UNITREE_ROS_DIR
    UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
    
  • 【可选】：如果要使用urdf文件，请修改robot_cfg.spawn

• 通过以下方式验证环境是否正确安装：

  • 列出可用任务：
    ./unitree_rl_lab.sh -l # 比isaaclab更快的版本
    
  • 运行任务：
    ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity # 支持任务名自动补全
    # 等同于
    python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
    
  • 使用训练好的智能体进行推理：
    ./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity # 支持任务名自动补全
    # 等同于
    python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
    

部署

模型训练完成后，我们需要在Mujoco中对训练好的策略进行sim2sim测试，然后部署sim2real。

环境设置

# 安装依赖
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# 安装unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # 安装到/usr/local目录
sudo make install
# 编译机器人控制器
cd unitree_rl_lab/deploy/robots/g1_29dof # 或其他机器人
mkdir build && cd build
cmake .. && make


Sim2Sim

安装https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation。

• 在/simulate/config.yaml中将robot设置为g1

• 将domain_id设置为0

• 将enable_elastic_hand设置为1

• 将use_joystck设置为1
# 启动仿真
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # 替代方案

cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. 按下[L2 + 上]键让机器人站立
# 2. 点击mujoco窗口，然后按8键让机器人脚部接触地面
# 3. 按下[R1 + X]键运行策略
# 4. 点击mujoco窗口，然后按9键禁用弹性带


Sim2Real

您可以直接使用此程序控制机器人，但请确保已关闭板载控制程序。
./g1_ctrl --network eth0 # eth0是网络接口名称


致谢

本仓库基于以下开源项目的支持和贡献构建。特别感谢：

• https://github.com/isaac-sim/IsaacLab：训练和运行代码的基础框架

• https://github.com/google-deepmind/mujoco.git：提供强大的仿真功能

• https://github.com/fan-ziqi/robot_lab：参考了项目结构和部分实现

• https://github.com/HybridRobotics/whole_body_tracking：多功能人形机器人运动跟踪控制框架