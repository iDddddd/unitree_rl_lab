#pragma once

#include <Eigen/Dense>
#include "data_bus.h"

#define  NUM_LEG 4
#define PROCESS_NOISE_PIMU   0.01
#define PROCESS_NOISE_VIMU   0.01
#define PROCESS_NOISE_PFOOT  0.01
#define SENSOR_NOISE_PIMU_REL_FOOT  0.001
#define SENSOR_NOISE_VIMU_REL_FOOT  0.001
#define SENSOR_NOISE_ZFOOT   0.001

class EKF {

    private:
   
        Eigen::Matrix<double, 18, 1> x; // estimation state
        Eigen::Matrix<double, 18, 1> xbar; // estimation state after process update
        Eigen::Matrix<double, 18, 18> P; // estimation state covariance
        Eigen::Matrix<double, 18, 18> Pbar; // estimation state covariance after process update
        Eigen::Matrix<double, 18, 18> A; // estimation state transition
        Eigen::Matrix<double, 18, 3> B; // estimation state transition
        Eigen::Matrix<double, 18, 18> Q; // estimation state transition noise

        Eigen::Matrix<double, 28, 1> y; //  observation
        Eigen::Matrix<double, 28, 1> yhat; // estimated observation
        Eigen::Matrix<double, 28, 1> error_y; // estimated observation
        Eigen::Matrix<double, 28, 1> Serror_y; // S^-1*error_y
        //C 其实就是论文里的 H
        Eigen::Matrix<double, 28, 18> C; // estimation state observation
        Eigen::Matrix<double, 28, 18> SC; // S^-1*C
        Eigen::Matrix<double, 28, 28> R_; // estimation state observation noise
        // helper matrices
        Eigen::Matrix<double, 3, 3> eye3; // 3x3 identity
        Eigen::Matrix<double, 28, 28> S_; // Innovation (or pre-fit residual) covariance
        Eigen::Matrix<double, 18, 28> K; // kalman gain

    public:
        void BasicEKF();
        void BasicEKF_(bool assume_flat_ground_);
        void init_state();



        double  estimated_root_pos[3];  //机身质心全局位置
        double  estimated_root_vel[3];  //机身质心全局速度

        Eigen::Matrix<double, 1, 3> root_pos ;   
        Eigen::Matrix<double, 1, 3> root_lin_vel ;  
        Eigen::Matrix<double, 3, 4> Touch_footposition;  

        bool x_estimated_contacts[4];
        bool assume_flat_ground = false ;
        bool filter_initialized = false ;

        // variables to process foot force
        double smooth_foot_force[4];
        double estimated_contacts[4];

        double dt = 0.001; 
        Eigen::Matrix3d root_rot_mat;
        Eigen::Vector3d imu_acc;
        Eigen::Matrix<double, 3, 4> foot_pos_rel;
        Eigen::Vector3d imu_ang_vel;
        Eigen::Matrix<double, 3, 4> foot_vel_rel;
        int movement_mode;
        Eigen::Vector4d foot_force;

        // DataBus::MotionState motionState;
        void update_dt(double dtin);
        void update_estimation();
        
        void dataBusRead(DataBus &robotState);
        void dataBusWrite(DataBus &robotState);
  
};
Eigen::Matrix3d skew(Eigen::Vector3d vec);
