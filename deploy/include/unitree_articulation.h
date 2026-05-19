// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/assets/articulation/articulation.h"
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/IMUState_.hpp>
#include <mutex>

namespace unitree
{

template <typename LowStatePtr>
class BaseArticulation : public isaaclab::Articulation
{
public:
    BaseArticulation(LowStatePtr lowstate_)
    : lowstate(lowstate_)
    {
        data.joystick = &lowstate->joystick;

        // Subscribe to foot IMU topics published by the MuJoCo bridge.
        // Each topic uses unitree_hg::msg::dds_::IMUState_ with:
        //   quaternion  – world frame [w,x,y,z]
        //   gyroscope   – body-frame angular velocity  (rad/s)
        //   accelerometer – body-frame specific force  (m/s²)
        //   temperature – contact normal force × 100   (N, int16_t)
        using IMUMsg = unitree_hg::msg::dds_::IMUState_;
        foot_imu_sub_[0] = std::make_shared<robot::ChannelSubscriber<IMUMsg>>(
            "rt/lf/foot_imu_0",
            [this](const void* msg) {
                const auto& m = *static_cast<const IMUMsg*>(msg);
                std::lock_guard<std::mutex> lk(foot_imu_mutex_);
                auto& d = data.foot_imu[0];
                d.quat_w    = Eigen::Quaternionf(m.quaternion()[0], m.quaternion()[1],
                                                  m.quaternion()[2], m.quaternion()[3]);
                d.ang_vel_b = Eigen::Vector3f(m.gyroscope()[0], m.gyroscope()[1], m.gyroscope()[2]);
                d.lin_acc_b = Eigen::Vector3f(m.accelerometer()[0], m.accelerometer()[1], m.accelerometer()[2]);
                d.contact_force = static_cast<float>(m.temperature()) / 100.0f;
            });
        foot_imu_sub_[1] = std::make_shared<robot::ChannelSubscriber<IMUMsg>>(
            "rt/lf/foot_imu_1",
            [this](const void* msg) {
                const auto& m = *static_cast<const IMUMsg*>(msg);
                std::lock_guard<std::mutex> lk(foot_imu_mutex_);
                auto& d = data.foot_imu[1];
                d.quat_w    = Eigen::Quaternionf(m.quaternion()[0], m.quaternion()[1],
                                                  m.quaternion()[2], m.quaternion()[3]);
                d.ang_vel_b = Eigen::Vector3f(m.gyroscope()[0], m.gyroscope()[1], m.gyroscope()[2]);
                d.lin_acc_b = Eigen::Vector3f(m.accelerometer()[0], m.accelerometer()[1], m.accelerometer()[2]);
                d.contact_force = static_cast<float>(m.temperature()) / 100.0f;
            });
        foot_imu_sub_[0]->InitChannel();
        foot_imu_sub_[1]->InitChannel();
    }

    void update() override
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        // base_angular_velocity
        for(int i(0); i<3; i++) {
            data.root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];
        }
        // base linear acceleration (specific force from base IMU)
        for(int i(0); i<3; i++) {
            data.root_lin_acc_b[i] = lowstate->msg_.imu_state().accelerometer()[i];
        }
        // project_gravity_body
        data.root_quat_w = Eigen::Quaternionf(
            lowstate->msg_.imu_state().quaternion()[0],
            lowstate->msg_.imu_state().quaternion()[1],
            lowstate->msg_.imu_state().quaternion()[2],
            lowstate->msg_.imu_state().quaternion()[3]
        );
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        // joint positions and velocities
        for(int i(0); i< (int)data.joint_ids_map.size(); i++) {
            data.joint_pos[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].q();
            data.joint_vel[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].dq();
        }
        // NOTE: foot_imu data is updated asynchronously via DDS callbacks above.
    }

    LowStatePtr lowstate;

private:
    using IMUSub = robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::IMUState_>;
    IMUSub foot_imu_sub_[2];
    std::mutex foot_imu_mutex_;
};

}