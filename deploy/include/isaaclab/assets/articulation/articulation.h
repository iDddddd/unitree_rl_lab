// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include "unitree/dds_wrapper/common/unitree_joystick.hpp"

namespace isaaclab
{

class MotionLoader;

// Foot IMU data received from DDS (rt/lf/foot_imu_0 / rt/lf/foot_imu_1).
// On the sim side the bridge packs contact normal force into temperature * 100.
struct FootImuState
{
    // World-frame orientation quaternion: [w, x, y, z]
    Eigen::Quaternionf quat_w       = Eigen::Quaternionf::Identity();
    // Body-frame angular velocity (rad/s)
    Eigen::Vector3f    ang_vel_b    = Eigen::Vector3f::Zero();
    // Body-frame specific force (m/s²), IMU convention (a_true + g)
    Eigen::Vector3f    lin_acc_b    = Eigen::Vector3f::Zero();
    // Contact normal force (N), decoded from DDS temperature field / 100.0
    float              contact_force = 0.0f;
};

struct ArticulationData
{
    Eigen::Vector3f GRAVITY_VEC_W = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    Eigen::Vector3f FORWARD_VEC_B = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

    std::vector<float> joint_stiffness; // sdk order
    std::vector<float> joint_damping; // sdk order

    // Joint positions of all joints.
    Eigen::VectorXf joint_pos;
    
    // Default joint positions of all joints.
    Eigen::VectorXf default_joint_pos;

    // Joint velocities of all joints.
    Eigen::VectorXf joint_vel;

    // Root angular velocity in base world frame.
    Eigen::Vector3f root_ang_vel_b;

    // Projection of the gravity direction on base frame.
    Eigen::Vector3f projected_gravity_b;

    Eigen::Quaternionf root_quat_w;

    // Base linear acceleration in body frame (from base IMU, specific force).
    Eigen::Vector3f root_lin_acc_b = Eigen::Vector3f::Zero();

    std::vector<float> joint_ids_map;

    unitree::common::UnitreeJoystick* joystick = nullptr;

    // Foot IMU data: index 0 = left ankle_roll_link, index 1 = right ankle_roll_link
    FootImuState foot_imu[2];
};

class Articulation
{
public:
    Articulation(){}

    virtual void update(){};

    ArticulationData data;
};

};