#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "platform_ekf.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        // TODO: smooth and limit the velocity commands
        cmd = key_commands[key];
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // If the deploy config contains EKF observations, bind the platform EKF.
    // The EKF instance is shared between the reset and step hooks.
    const auto& obs_cfg = env->cfg["observations"];
    bool needs_ekf = obs_cfg["ekf_base_vel_z_rel_platform"] ||
                     obs_cfg["ekf_base_roll_pitch_rel_platform"];
    if (needs_ekf) {
        auto ekf = std::make_shared<PlatformEKF>();
        auto robot_ptr = env->robot;

        // Helper: convert ArticulationData::foot_imu into FootImuData array
        auto make_foot_imu = [robot_ptr]() -> std::array<FootImuData, 2> {
            std::array<FootImuData, 2> fd;
            for (int i = 0; i < 2; ++i) {
                const auto& src = robot_ptr->data.foot_imu[i];
                fd[i].quat_w             = src.quat_w;
                fd[i].ang_vel_b          = src.ang_vel_b;
                fd[i].lin_acc_b          = src.lin_acc_b;
                fd[i].contact_normal_force = src.contact_force;
            }
            return fd;
        };

        env->ekf_reset_hook = [ekf, robot_ptr, make_foot_imu]() {
            auto fd = make_foot_imu();
            ekf->reset(robot_ptr->data.root_quat_w,
                       fd.data(),
                       robot_ptr->data.joint_pos.data(),
                       robot_ptr->data.joint_vel.data());
        };

        env->ekf_step_hook = [ekf, robot_ptr, make_foot_imu, env_ptr = env.get()]() {
            auto fd = make_foot_imu();
            auto out = ekf->step(robot_ptr->data.root_quat_w,
                                 robot_ptr->data.root_lin_acc_b,
                                 robot_ptr->data.root_ang_vel_b,
                                 fd.data(),
                                 robot_ptr->data.joint_pos.data(),
                                 robot_ptr->data.joint_vel.data(),
                                 env_ptr->step_dt);
            // Manually copy fields: ::PlatformEKFOutput -> isaaclab::PlatformEKFOutput
            env_ptr->ekf_output.base_vel_z_rel_platform  = out.base_vel_z_rel_platform;
            env_ptr->ekf_output.base_roll_rel_platform   = out.base_roll_rel_platform;
            env_ptr->ekf_output.base_pitch_rel_platform  = out.base_pitch_rel_platform;
        };
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}