// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// C++ port of PlatformRelativeEKF (source/…/estimators.py) for single-env
// deployment on real G1-29DOF hardware and sim2sim.
//
// State vector x (dim=12):
//   x[0:3]  – base position  relative to platform frame
//   x[3:6]  – base velocity  relative to platform frame
//   x[6:9]  – left  ankle-roll anchor position in platform frame
//   x[9:12] – right ankle-roll anchor position in platform frame
//
// Inputs each step:
//   base_imu  – from lowstate.imu_state()  (torso_link sensor, secondary_imu)
//   foot_imu  – from rt/lf/foot_imu_0 / rt/lf/foot_imu_1 (ankle_roll_link)
//   joint_pos / joint_vel – from lowstate.motor_state()

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>

// ---------------------------------------------------------------------------
// Sensor data snapshots (filled by the articulation layer)
// ---------------------------------------------------------------------------

struct FootImuData
{
    // World-frame orientation quaternion: [w, x, y, z] (IsaacLab / MuJoCo convention)
    Eigen::Quaternionf quat_w   = Eigen::Quaternionf::Identity();
    // Body-frame angular velocity and linear acceleration (specific force)
    Eigen::Vector3f ang_vel_b   = Eigen::Vector3f::Zero();
    Eigen::Vector3f lin_acc_b   = Eigen::Vector3f::Zero();
    // Contact normal force (N), decoded from temperature field (/100)
    float contact_normal_force  = 0.0f;
};

// ---------------------------------------------------------------------------
// EKF output (only the fields consumed by the policy observations)
// ---------------------------------------------------------------------------

struct PlatformEKFOutput
{
    // Base velocity in platform frame, z component only
    float base_vel_z_rel_platform       = 0.0f;
    // Roll and pitch of base relative to platform (from quat_rel)
    float base_roll_rel_platform        = 0.0f;
    float base_pitch_rel_platform       = 0.0f;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace ekf_detail
{

inline Eigen::Quaternionf normalize_quat(const Eigen::Quaternionf& q)
{
    float n = q.norm();
    if (n < 1.0e-9f) return Eigen::Quaternionf::Identity();
    return Eigen::Quaternionf(q.coeffs() / n);
}

// Apply quaternion to vector: rotates v by q
inline Eigen::Vector3f quat_apply(const Eigen::Quaternionf& q, const Eigen::Vector3f& v)
{
    return q * v;
}

// Apply inverse quaternion to vector
inline Eigen::Vector3f quat_apply_inv(const Eigen::Quaternionf& q, const Eigen::Vector3f& v)
{
    return q.conjugate() * v;
}

// Integrate world-frame angular velocity into a quaternion
// delta = ang_vel_w * dt, build axis-angle delta_quat, then q_new = delta_q * q_old
inline Eigen::Quaternionf integrate_quat(const Eigen::Quaternionf& q_w,
                                         const Eigen::Vector3f& omega_w,
                                         float dt)
{
    Eigen::Vector3f d = omega_w * dt;
    float angle = d.norm();
    if (angle < 1.0e-9f) return q_w;
    Eigen::Vector3f axis = d / angle;
    Eigen::Quaternionf dq(Eigen::AngleAxisf(angle, axis));
    // quat_unique: pick canonical sign (w >= 0)
    Eigen::Quaternionf result = normalize_quat(dq * q_w);
    if (result.w() < 0.0f) result.coeffs() = -result.coeffs();
    return result;
}

// Weighted average of two quaternions (sign-aligned to first)
inline Eigen::Quaternionf weighted_avg_quat(const Eigen::Quaternionf& q0,
                                            const Eigen::Quaternionf& q1,
                                            float w0, float w1,
                                            const Eigen::Quaternionf& fallback)
{
    float total = w0 + w1;
    if (total < 1.0e-6f) return fallback;
    Eigen::Quaternionf qa = q0;
    Eigen::Quaternionf qb = q1;
    // align signs
    if (qa.dot(qb) < 0.0f) qb.coeffs() = -qb.coeffs();
    Eigen::Vector4f sum = qa.coeffs() * (w0 / total) + qb.coeffs() * (w1 / total);
    float n = sum.norm();
    if (n < 1.0e-6f) return fallback;
    Eigen::Quaternionf res(sum / n);
    if (res.w() < 0.0f) res.coeffs() = -res.coeffs();
    return res;
}

// Roll-pitch-yaw from quaternion [w,x,y,z]
inline void quat_to_rpy(const Eigen::Quaternionf& q, float& roll, float& pitch, float& yaw)
{
    float w = q.w(), x = q.x(), y = q.y(), z = q.z();
    roll  = std::atan2(2.0f*(w*x + y*z),  1.0f - 2.0f*(x*x + y*y));
    float sinp = 2.0f*(w*y - z*x);
    sinp = std::clamp(sinp, -1.0f, 1.0f);
    pitch = std::asin(sinp);
    yaw   = std::atan2(2.0f*(w*z + x*y),  1.0f - 2.0f*(y*y + z*z));
}

inline float safe_clamp(float v, float abs_max)
{
    if (abs_max <= 0.0f) return v;
    if (!std::isfinite(v)) return 0.0f;
    return std::clamp(v, -abs_max, abs_max);
}

inline Eigen::Vector3f sanitize(const Eigen::Vector3f& v, float abs_max = -1.0f)
{
    Eigen::Vector3f out = v;
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(out[i])) out[i] = 0.0f;
        if (abs_max > 0.0f) out[i] = std::clamp(out[i], -abs_max, abs_max);
    }
    return out;
}

} // namespace ekf_detail

// ---------------------------------------------------------------------------
// FK helpers – direct port of _compute_foot_kinematics_fk from estimators.py
// ---------------------------------------------------------------------------

namespace g1_fk
{

struct LegFK
{
    Eigen::Vector3f pos;       // foot position in torso frame
    Eigen::Vector3f lin_vel;   // foot linear velocity in torso frame
};

// Apply fixed offset + optional RPY rotation
static void apply_fixed(Eigen::Vector3f& pos, Eigen::Matrix3f& rot,
                         Eigen::Vector3f& vel, Eigen::Vector3f& ang_vel,
                         float tx, float ty, float tz,
                         float roll = 0.0f, float pitch = 0.0f, float yaw = 0.0f)
{
    Eigen::Vector3f xyz(tx, ty, tz);
    Eigen::Vector3f offset = rot * xyz;
    pos    += offset;
    vel    += ang_vel.cross(offset);
    if (roll != 0.0f || pitch != 0.0f || yaw != 0.0f) {
        // Rzyx = Rz * Ry * Rx
        auto Rx = Eigen::AngleAxisf(roll,  Eigen::Vector3f::UnitX()).toRotationMatrix();
        auto Ry = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()).toRotationMatrix();
        auto Rz = Eigen::AngleAxisf(yaw,   Eigen::Vector3f::UnitZ()).toRotationMatrix();
        rot = rot * (Rz * Ry * Rx);
    }
}

// Apply revolute joint (axis in body frame before translation)
static void apply_revolute(Eigen::Matrix3f& rot, Eigen::Vector3f& vel,
                            Eigen::Vector3f& ang_vel,
                            const Eigen::Vector3f& axis_local,
                            float q, float qd)
{
    Eigen::Vector3f axis_w = rot * axis_local;
    ang_vel += axis_w * qd;
    rot = rot * Eigen::AngleAxisf(q, axis_local).toRotationMatrix();
}

// Apply URDF joint (translation + RPY + revolute)
static void apply_urdf_joint(Eigen::Vector3f& pos, Eigen::Matrix3f& rot,
                              Eigen::Vector3f& vel, Eigen::Vector3f& ang_vel,
                              float tx, float ty, float tz,
                              float rx, float ry, float rz,
                              const Eigen::Vector3f& axis,
                              float q, float qd)
{
    apply_fixed(pos, rot, vel, ang_vel, tx, ty, tz, rx, ry, rz);
    apply_revolute(rot, vel, ang_vel, axis, q, qd);
}

// Joint index helper – indices into the 29-DOF joint array ordered exactly as
// in PlatformVelocity deploy.yaml joint_ids_map:
//   [0..11] = left/right hip/knee/ankle (interleaved), [12..14] = waist, [15..28] arms
//
// In estimators.py the FK uses the TRAINING joint order (by joint name).
// Here we replicate the same mapping used in deploy (joint_ids_map).
//
// joint_ids_map = [0,6,12, 1,7,13, 2,8,14, 3,9,15, 22,4,10,16,23, 5,11,17,24, 18,25,19,26,20,27,21,28]
// But the FK only needs the leg/waist joints.  We store deploy-order indices:

// deploy joint ordering (0-indexed, matching joint_ids_map):
//   0: left_hip_pitch   1: left_hip_roll   2: left_hip_yaw
//   3: left_knee        4: left_ankle_pitch 5: left_ankle_roll
//   6: right_hip_pitch  7: right_hip_roll   8: right_hip_yaw
//   9: right_knee      10: right_ankle_pitch 11: right_ankle_roll
//  12: waist_yaw       13: waist_roll       14: waist_pitch

static constexpr int IDX_LEFT_HIP_PITCH    = 0;
static constexpr int IDX_LEFT_HIP_ROLL     = 1;
static constexpr int IDX_LEFT_HIP_YAW      = 2;
static constexpr int IDX_LEFT_KNEE         = 3;
static constexpr int IDX_LEFT_ANK_PITCH    = 4;
static constexpr int IDX_LEFT_ANK_ROLL     = 5;
static constexpr int IDX_RIGHT_HIP_PITCH   = 6;
static constexpr int IDX_RIGHT_HIP_ROLL    = 7;
static constexpr int IDX_RIGHT_HIP_YAW     = 8;
static constexpr int IDX_RIGHT_KNEE        = 9;
static constexpr int IDX_RIGHT_ANK_PITCH   = 10;
static constexpr int IDX_RIGHT_ANK_ROLL    = 11;
static constexpr int IDX_WAIST_YAW         = 12;
static constexpr int IDX_WAIST_ROLL        = 13;
static constexpr int IDX_WAIST_PITCH       = 14;

// Compute foot FK for one leg.
// side: 0 = left, 1 = right
// joint_pos/vel: pointer to the 29-DOF deploy-ordered array
LegFK compute_leg(int side,
                  const float* joint_pos,
                  const float* joint_vel)
{
    Eigen::Vector3f pos   = Eigen::Vector3f::Zero();
    Eigen::Matrix3f rot   = Eigen::Matrix3f::Identity();
    Eigen::Vector3f vel   = Eigen::Vector3f::Zero();
    Eigen::Vector3f omega = Eigen::Vector3f::Zero();

    // -----------------------------------------------------------------------
    // Inverse waist chain: torso_link → pelvis
    // (Same as Python: apply joints in reverse with negated angles)
    // -----------------------------------------------------------------------
    float wpit  = joint_pos[IDX_WAIST_PITCH];
    float wroll = joint_pos[IDX_WAIST_ROLL];
    float wyaw  = joint_pos[IDX_WAIST_YAW];
    float wpit_d  = joint_vel[IDX_WAIST_PITCH];
    float wroll_d = joint_vel[IDX_WAIST_ROLL];
    float wyaw_d  = joint_vel[IDX_WAIST_YAW];

    // waist_pitch inverse (subtract offset from torso → waist_roll origin)
    apply_revolute(rot, vel, omega, Eigen::Vector3f(0.0f, 1.0f, 0.0f), -wpit, -wpit_d);
    apply_revolute(rot, vel, omega, Eigen::Vector3f(1.0f, 0.0f, 0.0f), -wroll, -wroll_d);
    // Fixed offset: torso → waist_roll origin (URDF: waist_roll child pos relative to waist_yaw)
    apply_fixed(pos, rot, vel, omega, 0.0039635f, 0.0f, -0.044f);
    apply_revolute(rot, vel, omega, Eigen::Vector3f(0.0f, 0.0f, 1.0f), -wyaw, -wyaw_d);

    // -----------------------------------------------------------------------
    // Now at pelvis origin; descend leg chain
    // -----------------------------------------------------------------------
    float ys = (side == 0) ? 1.0f : -1.0f;  // left: +y, right: -y

    int hip_pitch_idx   = (side == 0) ? IDX_LEFT_HIP_PITCH  : IDX_RIGHT_HIP_PITCH;
    int hip_roll_idx    = (side == 0) ? IDX_LEFT_HIP_ROLL   : IDX_RIGHT_HIP_ROLL;
    int hip_yaw_idx     = (side == 0) ? IDX_LEFT_HIP_YAW    : IDX_RIGHT_HIP_YAW;
    int knee_idx        = (side == 0) ? IDX_LEFT_KNEE       : IDX_RIGHT_KNEE;
    int ankle_pitch_idx = (side == 0) ? IDX_LEFT_ANK_PITCH  : IDX_RIGHT_ANK_PITCH;
    int ankle_roll_idx  = (side == 0) ? IDX_LEFT_ANK_ROLL   : IDX_RIGHT_ANK_ROLL;

    // hip_pitch: (0, ±0.064452, -0.1027), RPY=(0,0,0), axis=(0,1,0)
    apply_urdf_joint(pos, rot, vel, omega,
                     0.0f, ys*0.064452f, -0.1027f,
                     0.0f, 0.0f, 0.0f,
                     Eigen::Vector3f(0.0f, 1.0f, 0.0f),
                     joint_pos[hip_pitch_idx], joint_vel[hip_pitch_idx]);

    // hip_roll: (0, ±0.052, -0.030465), RPY=(0,-0.1749,0), axis=(1,0,0)
    apply_urdf_joint(pos, rot, vel, omega,
                     0.0f, ys*0.052f, -0.030465f,
                     0.0f, -0.1749f, 0.0f,
                     Eigen::Vector3f(1.0f, 0.0f, 0.0f),
                     joint_pos[hip_roll_idx], joint_vel[hip_roll_idx]);

    // hip_yaw: (0.025001, 0, -0.12412), RPY=(0,0,0), axis=(0,0,1)
    apply_urdf_joint(pos, rot, vel, omega,
                     0.025001f, 0.0f, -0.12412f,
                     0.0f, 0.0f, 0.0f,
                     Eigen::Vector3f(0.0f, 0.0f, 1.0f),
                     joint_pos[hip_yaw_idx], joint_vel[hip_yaw_idx]);

    // knee: (-0.078273, ±0.0021489, -0.17734), RPY=(0,0.1749,0), axis=(0,1,0)
    apply_urdf_joint(pos, rot, vel, omega,
                     -0.078273f, ys*0.0021489f, -0.17734f,
                     0.0f, 0.1749f, 0.0f,
                     Eigen::Vector3f(0.0f, 1.0f, 0.0f),
                     joint_pos[knee_idx], joint_vel[knee_idx]);

    // ankle_pitch: (0, ∓9.4445e-5, -0.30001), RPY=(0,0,0), axis=(0,1,0)
    apply_urdf_joint(pos, rot, vel, omega,
                     0.0f, -ys*9.4445e-5f, -0.30001f,
                     0.0f, 0.0f, 0.0f,
                     Eigen::Vector3f(0.0f, 1.0f, 0.0f),
                     joint_pos[ankle_pitch_idx], joint_vel[ankle_pitch_idx]);

    // ankle_roll: (0, 0, -0.017558), RPY=(0,0,0), axis=(1,0,0)
    apply_urdf_joint(pos, rot, vel, omega,
                     0.0f, 0.0f, -0.017558f,
                     0.0f, 0.0f, 0.0f,
                     Eigen::Vector3f(1.0f, 0.0f, 0.0f),
                     joint_pos[ankle_roll_idx], joint_vel[ankle_roll_idx]);

    return {pos, vel};
}

} // namespace g1_fk

// ---------------------------------------------------------------------------
// PlatformEKF class
// ---------------------------------------------------------------------------

class PlatformEKF
{
public:
    // --- Tunable parameters (matching Python defaults) ---
    float contact_force_threshold     = 5.0f;   // N
    float contact_prob_smoothing      = 0.2f;
    float stance_prob_threshold       = 0.5f;
    float attitude_correction_gain    = 0.15f;

    float q_pos                       = 1.0e-3f;
    float q_vel                       = 1.5e-1f;
    float q_pos_air                   = 1.0e-2f;
    float q_vel_air                   = 5.0e-1f;
    float q_anchor_stance             = 1.0e-5f;
    float q_anchor_swing              = 5.0e-2f;
    float q_anchor_air                = 2.5e-1f;

    float r_foot_pos_stance           = 2.0e-4f;
    float r_foot_pos_swing            = 2.0e-1f;
    float r_foot_vel_stance           = 3.0e-2f;
    float r_foot_vel_swing            = 5.0e-1f;

    float init_cov                    = 1.0e-2f;
    float cov_jitter                  = 1.0e-6f;
    float max_abs_pos                 = 5.0f;
    float max_abs_vel                 = 10.0f;
    float max_abs_acc                 = 50.0f;
    float max_abs_ang_rate            = 25.0f;
    float ang_acc_smoothing           = 0.85f;
    float vel_output_smoothing        = 0.3f;
    float innovation_clip             = 1.0f;
    float touchdown_impact_damping    = 0.15f;

    PlatformEKF()
    {
        x_.setZero();
        P_ = Eigen::Matrix<float, 12, 12>::Identity() * init_cov;
        platform_quat_w_ = Eigen::Quaternionf::Identity();
        platform_ang_vel_w_.setZero();
        platform_ang_acc_w_.setZero();
        platform_lin_acc_w_.setZero();
        smoothed_vel_.setZero();
        for (int i = 0; i < 2; ++i) {
            contact_prob_[i] = 0.0f;
            prev_contact_prob_[i] = 0.0f;
        }
        initialized_ = false;
    }

    // Call when the episode resets.
    // joint_pos/vel: 29-DOF deploy-ordered arrays.
    void reset(const Eigen::Quaternionf& base_quat_w,
               const FootImuData foot_imu[2],
               const float* joint_pos,
               const float* joint_vel)
    {
        // Measure initial contacts
        for (int i = 0; i < 2; ++i) {
            float p = _measureContact(foot_imu[i].contact_normal_force);
            contact_prob_[i]      = p;
            prev_contact_prob_[i] = p;
        }

        // Fuse platform IMU
        _fusePlatformIMU(foot_imu, contact_prob_, nullptr);

        // Relative orientation
        Eigen::Quaternionf quat_rel = ekf_detail::normalize_quat(
            platform_quat_w_.conjugate() * base_quat_w);

        // Foot kinematics in base frame
        Eigen::Vector3f foot_pos_b[2], foot_vel_b[2];
        for (int i = 0; i < 2; ++i) {
            g1_fk::LegFK fk = g1_fk::compute_leg(i, joint_pos, joint_vel);
            // Rotate from torso frame to platform frame
            foot_pos_b[i] = ekf_detail::quat_apply(quat_rel, fk.pos);
            foot_vel_b[i] = ekf_detail::quat_apply(quat_rel, fk.lin_vel);
        }

        // Initialize base pos relative to platform from stance feet midpoint
        Eigen::Vector3f foot_pos_p[2] = {foot_pos_b[0], foot_pos_b[1]};
        Eigen::Vector3f base_pos_rel  = _initBasePos(foot_pos_p, contact_prob_);
        Eigen::Vector3f base_vel_rel  = _initBaseVel(foot_pos_p, foot_vel_b, contact_prob_);

        x_.segment<3>(0) = base_pos_rel;
        x_.segment<3>(3) = base_vel_rel;
        x_.segment<3>(6) = base_pos_rel + foot_pos_p[0];   // left  anchor
        x_.segment<3>(9) = base_pos_rel + foot_pos_p[1];   // right anchor

        P_ = Eigen::Matrix<float, 12, 12>::Identity() * init_cov;
        smoothed_vel_ = base_vel_rel;
        initialized_ = true;
    }

    // One EKF step.  Returns the observation needed by the policy.
    PlatformEKFOutput step(const Eigen::Quaternionf& base_quat_w,
                           const Eigen::Vector3f&    base_lin_acc_b,
                           const Eigen::Vector3f&    base_ang_vel_b,
                           const FootImuData         foot_imu[2],
                           const float*              joint_pos,
                           const float*              joint_vel,
                           float                     dt)
    {
        // ---- Contact probability update ----
        bool touchdown[2] = {false, false};
        for (int i = 0; i < 2; ++i) {
            float raw = _measureContact(foot_imu[i].contact_normal_force);
            float smooth = (1.0f - contact_prob_smoothing) * contact_prob_[i]
                         + contact_prob_smoothing * raw;
            smooth = std::clamp(smooth, 0.0f, 1.0f);
            bool was_stance = (prev_contact_prob_[i] > stance_prob_threshold);
            bool now_stance = (smooth > stance_prob_threshold);
            touchdown[i] = now_stance && !was_stance;
            prev_contact_prob_[i] = contact_prob_[i];
            contact_prob_[i] = smooth;
        }
        bool airborne = (contact_prob_[0] <= stance_prob_threshold) &&
                        (contact_prob_[1] <= stance_prob_threshold);

        // ---- Platform IMU fusion ----
        _fusePlatformIMU(foot_imu, contact_prob_, touchdown);
        Eigen::Quaternionf prop_quat = ekf_detail::integrate_quat(
            platform_quat_w_, platform_ang_vel_w_, dt);

        float w0 = 1.0f - attitude_correction_gain;
        float w1 = attitude_correction_gain
                 * std::clamp(contact_prob_[0] + contact_prob_[1], 0.0f, 1.0f);
        Eigen::Quaternionf fused_quat_sensor = _fusedQuat(foot_imu, contact_prob_);
        platform_quat_w_ = ekf_detail::weighted_avg_quat(
            prop_quat, fused_quat_sensor, w0, w1, prop_quat);

        // Smooth angular acceleration
        Eigen::Vector3f ang_acc_new = _fuseAngAcc(foot_imu, contact_prob_);
        platform_ang_acc_w_ = ang_acc_smoothing * platform_ang_acc_w_
                            + (1.0f - ang_acc_smoothing) * ang_acc_new;

        // ---- Kinematics ----
        Eigen::Quaternionf quat_rel = ekf_detail::normalize_quat(
            platform_quat_w_.conjugate() * base_quat_w);
        Eigen::Vector3f foot_pos_p[2], foot_vel_p[2];
        for (int i = 0; i < 2; ++i) {
            g1_fk::LegFK fk = g1_fk::compute_leg(i, joint_pos, joint_vel);
            foot_pos_p[i] = ekf_detail::quat_apply(quat_rel, fk.pos);
            foot_vel_p[i] = ekf_detail::quat_apply(quat_rel, fk.lin_vel);
        }

        // ---- EKF Prediction ----
        // Relative acceleration in platform frame (gravity-safe)
        Eigen::Vector3f base_lin_acc_w = ekf_detail::quat_apply(base_quat_w, base_lin_acc_b);
        Eigen::Vector3f diff_acc_w     = base_lin_acc_w - platform_lin_acc_w_;
        Eigen::Vector3f rel_acc_p      = ekf_detail::quat_apply_inv(platform_quat_w_, diff_acc_w);
        rel_acc_p = ekf_detail::sanitize(rel_acc_p, max_abs_acc);

        // Coriolis and centrifugal corrections
        Eigen::Vector3f omega_p = ekf_detail::quat_apply_inv(platform_quat_w_, platform_ang_vel_w_);
        omega_p = ekf_detail::sanitize(omega_p, max_abs_ang_rate);
        Eigen::Vector3f rel_vel_p = x_.segment<3>(3);
        Eigen::Vector3f rel_pos_p = x_.segment<3>(0);
        rel_acc_p -= 2.0f * omega_p.cross(rel_vel_p);
        rel_acc_p -= omega_p.cross(omega_p.cross(rel_pos_p));
        rel_acc_p = ekf_detail::sanitize(rel_acc_p, max_abs_acc);
        // Suppress vertical accel prediction (mirrors use_vertical_accel_prediction=False)
        rel_acc_p[2] = 0.0f;

        Eigen::Matrix<float, 12, 1> xbar = x_;
        xbar.segment<3>(0) += dt * rel_vel_p + 0.5f * dt * dt * rel_acc_p;
        xbar.segment<3>(3) += dt * rel_acc_p;
        xbar = _sanitizeState(xbar);

        // Transition matrix A
        Eigen::Matrix<float, 12, 12> A = Eigen::Matrix<float, 12, 12>::Identity();
        A.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity() * dt;
        Eigen::Matrix<float, 12, 12> Pbar = A * P_ * A.transpose()
                                           + _buildQ(dt, airborne);
        Pbar = _sanitizeCov(Pbar);

        // Touchdown anchor re-seed
        if (touchdown[0] || touchdown[1]) {
            for (int i = 0; i < 2; ++i) {
                if (touchdown[i]) {
                    xbar.segment<3>(6 + 3*i) = xbar.segment<3>(0) + foot_pos_p[i];
                }
            }
        }

        // ---- EKF Update (only when at least one foot in stance) ----
        bool has_stance = (contact_prob_[0] > stance_prob_threshold) ||
                          (contact_prob_[1] > stance_prob_threshold);
        if (has_stance) {
            static const Eigen::Matrix<float, 12, 12> H = _buildH();
            Eigen::Matrix<float, 12, 1> yhat = _predictMeasurement(
                xbar, foot_pos_p, foot_vel_p);
            Eigen::Matrix<float, 12, 1> innovation = -yhat;
            // Clip innovation
            if (innovation_clip > 0.0f) {
                for (int i = 0; i < 12; ++i) {
                    innovation[i] = std::clamp(innovation[i], -innovation_clip, innovation_clip);
                }
            }
            Eigen::Matrix<float, 12, 12> R = _buildR();
            Eigen::Matrix<float, 12, 12> S = H * Pbar * H.transpose() + R;
            S = 0.5f * (S + S.transpose());
            S += Eigen::Matrix<float, 12, 12>::Identity() * cov_jitter;

            if (_isFinite(S) && _isFinite(Pbar)) {
                Eigen::Matrix<float, 12, 12> PHt = Pbar * H.transpose();
                Eigen::Matrix<float, 12, 12> K = PHt * S.inverse();
                xbar += K * innovation;
                xbar = _sanitizeState(xbar);
                // Joseph-form covariance update
                Eigen::Matrix<float, 12, 12> IKH =
                    Eigen::Matrix<float, 12, 12>::Identity() - K * H;
                Pbar = IKH * Pbar * IKH.transpose() + K * R * K.transpose();
                Pbar = _sanitizeCov(Pbar);
            }
        }

        x_ = xbar;
        P_ = Pbar;

        // ---- Velocity smoothing ----
        Eigen::Vector3f raw_vel = x_.segment<3>(3);
        smoothed_vel_ = vel_output_smoothing * smoothed_vel_
                      + (1.0f - vel_output_smoothing) * raw_vel;

        // ---- Build output ----
        float roll, pitch, yaw_unused;
        ekf_detail::quat_to_rpy(quat_rel, roll, pitch, yaw_unused);

        PlatformEKFOutput out;
        out.base_vel_z_rel_platform   = ekf_detail::safe_clamp(smoothed_vel_[2], max_abs_vel);
        out.base_roll_rel_platform    = ekf_detail::safe_clamp(roll,  3.14159f);
        out.base_pitch_rel_platform   = ekf_detail::safe_clamp(pitch, 3.14159f);
        return out;
    }

    bool isInitialized() const { return initialized_; }

private:
    bool initialized_ = false;

    // EKF state and covariance
    Eigen::Matrix<float, 12, 1>   x_;
    Eigen::Matrix<float, 12, 12>  P_;

    // Platform state estimate
    Eigen::Quaternionf platform_quat_w_;
    Eigen::Vector3f    platform_ang_vel_w_;
    Eigen::Vector3f    platform_ang_acc_w_;
    Eigen::Vector3f    platform_lin_acc_w_;

    // Contact state
    float contact_prob_[2]      = {0.0f, 0.0f};
    float prev_contact_prob_[2] = {0.0f, 0.0f};

    // Smoothed velocity output
    Eigen::Vector3f smoothed_vel_;

    // ---- Contact probability ----
    float _measureContact(float normal_force) const
    {
        float f = std::max(0.0f, normal_force);
        if (f < contact_force_threshold) return 0.0f;
        // same as Python: weight = f_z / (|f| + eps), but we only have normal force
        return std::clamp(f / (f + 1.0e-6f), 0.0f, 1.0f);
    }

    // ---- Platform IMU fusion ----
    Eigen::Quaternionf _fusedQuat(const FootImuData foot_imu[2],
                                   const float contact_prob[2]) const
    {
        float w0 = contact_prob[0], w1 = contact_prob[1];
        if (w0 + w1 < 1.0e-6f) return platform_quat_w_;
        return ekf_detail::weighted_avg_quat(
            foot_imu[0].quat_w, foot_imu[1].quat_w,
            w0, w1, platform_quat_w_);
    }

    Eigen::Vector3f _fuseAngAcc(const FootImuData foot_imu[2],
                                 const float contact_prob[2]) const
    {
        // Approximate ang_acc as 0 in C++ (avoiding finite-diff noise on deploy).
        // The Python version also notes this is the main source of spikes.
        (void)foot_imu; (void)contact_prob;
        return Eigen::Vector3f::Zero();
    }

    void _fusePlatformIMU(const FootImuData foot_imu[2],
                           const float contact_prob[2],
                           const bool touchdown[2])
    {
        // Effective weights: dampen touchdown transients
        float eff[2];
        for (int i = 0; i < 2; ++i) {
            eff[i] = contact_prob[i];
            if (touchdown && touchdown[i])
                eff[i] *= touchdown_impact_damping;
        }
        float total = eff[0] + eff[1];
        bool has_contact = (total > 1.0e-6f);

        Eigen::Quaternionf q_fused =
            has_contact ? _fusedQuat(foot_imu, eff) : platform_quat_w_;

        // Fuse angular velocity
        Eigen::Vector3f ang_vel_fused = Eigen::Vector3f::Zero();
        if (has_contact) {
            float w0 = eff[0] / total, w1 = eff[1] / total;
            // Convert body-frame ang_vel to world frame using each foot's quat_w
            Eigen::Vector3f av0 = ekf_detail::quat_apply(foot_imu[0].quat_w, foot_imu[0].ang_vel_b);
            Eigen::Vector3f av1 = ekf_detail::quat_apply(foot_imu[1].quat_w, foot_imu[1].ang_vel_b);
            ang_vel_fused = w0 * av0 + w1 * av1;
        } else {
            ang_vel_fused = platform_ang_vel_w_;
        }
        // Fuse linear acceleration (world frame, specific force)
        Eigen::Vector3f lin_acc_fused = Eigen::Vector3f::Zero();
        if (has_contact) {
            float w0 = eff[0] / total, w1 = eff[1] / total;
            Eigen::Vector3f la0 = ekf_detail::quat_apply(foot_imu[0].quat_w, foot_imu[0].lin_acc_b);
            Eigen::Vector3f la1 = ekf_detail::quat_apply(foot_imu[1].quat_w, foot_imu[1].lin_acc_b);
            lin_acc_fused = w0 * la0 + w1 * la1;
        } else {
            lin_acc_fused = platform_lin_acc_w_;
        }

        platform_quat_w_    = q_fused;
        platform_ang_vel_w_ = ekf_detail::sanitize(ang_vel_fused, max_abs_ang_rate);
        platform_lin_acc_w_ = ekf_detail::sanitize(lin_acc_fused, max_abs_acc);
    }

    // ---- Initialization helpers ----
    Eigen::Vector3f _initBasePos(const Eigen::Vector3f foot_pos_p[2],
                                  const float contact_prob[2]) const
    {
        bool ls = contact_prob[0] > stance_prob_threshold;
        bool rs = contact_prob[1] > stance_prob_threshold;
        if (ls && rs)
            return -0.5f * (foot_pos_p[0] + foot_pos_p[1]);
        if (ls) return -foot_pos_p[0];
        if (rs) return -foot_pos_p[1];
        return -0.5f * (foot_pos_p[0] + foot_pos_p[1]);
    }

    Eigen::Vector3f _initBaseVel(const Eigen::Vector3f foot_pos_p[2],
                                  const Eigen::Vector3f foot_vel_p[2],
                                  const float contact_prob[2]) const
    {
        (void)foot_pos_p;
        Eigen::Vector3f vel = Eigen::Vector3f::Zero();
        float cnt = 0.0f;
        for (int i = 0; i < 2; ++i) {
            if (contact_prob[i] > stance_prob_threshold) {
                vel -= foot_vel_p[i];
                cnt += 1.0f;
            }
        }
        if (cnt > 0.0f) vel /= cnt;
        return vel;
    }

    // ---- Process noise Q ----
    Eigen::Matrix<float, 12, 12> _buildQ(float dt, bool airborne) const
    {
        Eigen::Matrix<float, 12, 12> Q;
        Q.setZero();
        float qp = airborne ? q_pos_air : q_pos;
        float qv = airborne ? q_vel_air : q_vel;
        Q.block<3,3>(0,0) = Eigen::Matrix3f::Identity() * qp * dt * dt;
        Q.block<3,3>(3,3) = Eigen::Matrix3f::Identity() * qv * dt;
        for (int i = 0; i < 2; ++i) {
            float cp = contact_prob_[i];
            float an = airborne ? q_anchor_air
                                : q_anchor_stance * cp + q_anchor_swing * (1.0f - cp);
            Q.block<3,3>(6+3*i, 6+3*i) = Eigen::Matrix3f::Identity() * an;
        }
        return Q;
    }

    // ---- Measurement matrix H (static) ----
    static Eigen::Matrix<float, 12, 12> _buildH()
    {
        Eigen::Matrix<float, 12, 12> H;
        H.setZero();
        Eigen::Matrix3f I3 = Eigen::Matrix3f::Identity();
        // Left foot position measurement: base_pos + foot_pos_b - left_anchor == 0
        H.block<3,3>(0,0) =  I3;
        H.block<3,3>(0,6) = -I3;
        // Left foot velocity measurement: base_vel + foot_vel_b == 0 (stance)
        H.block<3,3>(3,3) =  I3;
        // Right foot position
        H.block<3,3>(6,0) =  I3;
        H.block<3,3>(6,9) = -I3;
        // Right foot velocity
        H.block<3,3>(9,3) =  I3;
        return H;
    }

    // ---- Predicted measurement ----
    Eigen::Matrix<float, 12, 1> _predictMeasurement(
        const Eigen::Matrix<float, 12, 1>& x,
        const Eigen::Vector3f foot_pos_p[2],
        const Eigen::Vector3f foot_vel_p[2]) const
    {
        Eigen::Matrix<float, 12, 1> y;
        Eigen::Vector3f base_pos = x.segment<3>(0);
        Eigen::Vector3f base_vel = x.segment<3>(3);
        Eigen::Vector3f anch0    = x.segment<3>(6);
        Eigen::Vector3f anch1    = x.segment<3>(9);
        // y = predicted - 0 (zero innovation target)
        y.segment<3>(0) = base_pos + foot_pos_p[0] - anch0;
        y.segment<3>(3) = base_vel + foot_vel_p[0];
        y.segment<3>(6) = base_pos + foot_pos_p[1] - anch1;
        y.segment<3>(9) = base_vel + foot_vel_p[1];
        return y;
    }

    // ---- Measurement noise R ----
    Eigen::Matrix<float, 12, 12> _buildR() const
    {
        Eigen::Matrix<float, 12, 12> R;
        R.setZero();
        for (int i = 0; i < 2; ++i) {
            float cp = contact_prob_[i];
            float rp = r_foot_pos_stance * cp + r_foot_pos_swing * (1.0f - cp);
            float rv = r_foot_vel_stance * cp + r_foot_vel_swing * (1.0f - cp);
            R.block<3,3>(0 + 6*i, 0 + 6*i) = Eigen::Matrix3f::Identity() * rp;
            R.block<3,3>(3 + 6*i, 3 + 6*i) = Eigen::Matrix3f::Identity() * rv;
        }
        return R;
    }

    // ---- Sanitize / symmetrize ----
    Eigen::Matrix<float, 12, 1> _sanitizeState(
        const Eigen::Matrix<float, 12, 1>& s) const
    {
        Eigen::Matrix<float, 12, 1> out = s;
        for (int i = 0; i < 12; ++i) {
            if (!std::isfinite(out[i])) out[i] = 0.0f;
        }
        for (int i = 0; i < 3; ++i) out[i] = std::clamp(out[i], -max_abs_pos, max_abs_pos);
        for (int i = 3; i < 6; ++i) out[i] = std::clamp(out[i], -max_abs_vel, max_abs_vel);
        for (int i = 6; i < 12; ++i) out[i] = std::clamp(out[i], -max_abs_pos, max_abs_pos);
        return out;
    }

    Eigen::Matrix<float, 12, 12> _sanitizeCov(
        const Eigen::Matrix<float, 12, 12>& C) const
    {
        Eigen::Matrix<float, 12, 12> out = 0.5f * (C + C.transpose());
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 12; ++j) {
                if (!std::isfinite(out(i,j))) out(i,j) = 0.0f;
                out(i,j) = std::clamp(out(i,j), -1.0e6f, 1.0e6f);
            }
            out(i,i) = std::max(out(i,i), cov_jitter);
        }
        out += Eigen::Matrix<float, 12, 12>::Identity() * cov_jitter;
        return out;
    }

    static bool _isFinite(const Eigen::Matrix<float, 12, 12>& M)
    {
        return M.array().isFinite().all();
    }
};
