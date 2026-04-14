#include "EKF.h"

void EKF::BasicEKF ()
{
    eye3.setIdentity();
    C.setZero();
    for (int i=0; i<NUM_LEG; ++i) {
        C.block<3,3>(i*3,0) = -eye3;  //-pos    与论文中的相反
        C.block<3,3>(i*3,6+i*3) = eye3;  //foot pos 与论文中的相反
        C.block<3,3>(NUM_LEG*3+i*3,3) = eye3;  // vel
        C(NUM_LEG*6+i,6+i*3+2) = 1;  // height z of foot
    }
    //Q状态转移协方差矩阵
    Q.setIdentity();
    Q.block<3,3>(0,0) = PROCESS_NOISE_PIMU*eye3;               // position transition
    Q.block<3,3>(3,3) = PROCESS_NOISE_VIMU*eye3;               // velocity transition
    for (int i=0; i<NUM_LEG; ++i) {
        Q.block<3,3>(6+i*3,6+i*3) = PROCESS_NOISE_PFOOT*eye3;  // foot position transition
    }
    R_.setIdentity();
    for (int i=0; i<NUM_LEG; ++i) {
        R_.block<3,3>(i*3,i*3) = SENSOR_NOISE_PIMU_REL_FOOT*eye3;                        // fk estimation
        R_.block<3,3>(NUM_LEG*3+i*3,NUM_LEG*3+i*3) = SENSOR_NOISE_VIMU_REL_FOOT*eye3;      // vel estimation
        R_(NUM_LEG*6+i,NUM_LEG*6+i) = SENSOR_NOISE_ZFOOT;                               // height z estimation
    }
    A.setIdentity();
    B.setZero();
    assume_flat_ground = true;
}


void EKF::BasicEKF_(bool assume_flat_ground_)
{
       assume_flat_ground = assume_flat_ground_;
    if (assume_flat_ground == false) {
        for (int i=0; i<NUM_LEG; ++i) {
            R_(NUM_LEG*6+i,NUM_LEG*6+i) = 1e5;   // height z estimation not reliable
        }
    }
}


void EKF:: init_state()
{
    filter_initialized = true;
    P.setIdentity();
    P = P * 3;
    // set initial value of x
    // x.setZero();
    x.segment<3>(0) = Eigen::Vector3d(0, 0, 1.01);//位置
    x.segment<3>(3) = Eigen::Vector3d(0, 0, 0);  //速度
    x.segment<3>(6) = Eigen::Vector3d(-0.0, 0.116, 0);  //左腿
    x.segment<3>(9) = Eigen::Vector3d(-0.0, -0.116, 0);  //右腿
    x.segment<3>(12) = Eigen::Vector3d(-0.0, 0.116, 0);  //左腿
    x.segment<3>(15) = Eigen::Vector3d(-0.0, -0.116, 0);  //右腿
    // for (int i = 0; i < NUM_LEG; ++i) {
    //     Eigen::Vector3d fk_pos = foot_pos_rel.block<3, 1>(0, i);
    //     x.segment<3>(6 + i * 3) = root_rot_mat * fk_pos + x.segment<3>(0);
    // }
}

void EKF::update_estimation()
{
    
    EKF::BasicEKF(); //卡尔曼滤波矩阵赋值
    // update A B using latest dt
    A.block<3, 3>(0, 3) = dt * eye3;
    B.block<3, 3>(3, 0) = dt * eye3;

    // control input u is Ra + ag
    Eigen::Vector3d u = root_rot_mat * imu_acc + Eigen::Vector3d(0, 0, -9.81);
    std::cout << "观测出来的加速度：" << u.transpose() << std::endl;
    // contact estimation, do something very simple first
    if (movement_mode == 0) {  // stand
        for (int i = 0; i < NUM_LEG; ++i) estimated_contacts[i] = 1.0;
    } else {  // walk
        for (int i = 0; i < NUM_LEG; ++i) {
            estimated_contacts[i] = std::min(std::max((foot_force(i)) / (450.0 - 0.0), 0.0), 1.0);
//        estimated_contacts[i] = 1.0/(1.0+std::exp(-(state.foot_force(i)-100)));
        }
    }
    // update Q
    Q.block<3, 3>(0, 0) = PROCESS_NOISE_PIMU * dt / 20.0 * eye3;
    Q.block<3, 3>(3, 3) = PROCESS_NOISE_VIMU * dt * 9.8 / 20.0 * eye3;
    // update Q R for legs not in contact
    for (int i = 0; i < NUM_LEG; ++i) {
        Q.block<3, 3>(6 + i * 3, 6 + i * 3)
                =
                (1 + (1 - estimated_contacts[i]) * 1e10) * dt * PROCESS_NOISE_PFOOT * eye3;  // foot position transition
        // for estimated_contacts[i] == 1, Q = 0.002   确定接触
        // for estimated_contacts[i] == 0, Q = 1001*Q   不确定接触

        R_.block<3, 3>(i * 3, i * 3)
                = (1 + (1 - estimated_contacts[i]) * 1e10) * SENSOR_NOISE_PIMU_REL_FOOT *
                  eye3;                       // fk estimation

        R_.block<3, 3>(NUM_LEG * 3 + i * 3, NUM_LEG * 3 + i * 3)
                = (1 + (1 - estimated_contacts[i]) * 1e10) * SENSOR_NOISE_VIMU_REL_FOOT * eye3;      // vel estimation
        if (assume_flat_ground) {
            R_(NUM_LEG * 6 + i, NUM_LEG * 6 + i)
                    = (1 + (1 - estimated_contacts[i]) * 1e10) * SENSOR_NOISE_ZFOOT;       // height z estimation
        }
    }
    // process update
    //卡尔曼滤波器的离散状态方程
    xbar = A * x + B * u;
    Pbar = A * P * A.transpose() + Q;

    // measurement construction
    //卡尔曼滤波器的观测方程
    yhat = C * xbar;

    // actual measurement   zk= H * xk
    for (int i=0; i<NUM_LEG; ++i) {
        Eigen::Vector3d fk_pos = foot_pos_rel.block<3,1>(0,i);
        y.block<3,1>(i*3,0) = root_rot_mat * fk_pos;   // fk estimation 足端位置相对于质心
        Eigen::Vector3d leg_v = -foot_vel_rel.block<3,1>(0,i) - 
                  skew(imu_ang_vel) * fk_pos;
        y.block<3,1>(NUM_LEG*3+i*3,0) =
                (1.0-estimated_contacts[i])* x.segment<3>(3) +  estimated_contacts[i]*root_rot_mat*leg_v;      // vel estimation

        y(NUM_LEG*6+i) =
                (1.0-estimated_contacts[i])*(x(2) + (root_rot_mat * fk_pos)(2)) 
                   + estimated_contacts[i]*0;                               // height z estimation
    }

    S_ = C * Pbar *C.transpose() + R_;
    //不能理解//
    S_ = 0.5*(S_+S_.transpose());
     
    error_y = y - yhat;
    // Serror_y = S_.fullPivHouseholderQr().solve(error_y);

    Serror_y = S_.lu().solve(error_y);
    x = xbar + Pbar * C.transpose() * Serror_y;

    SC = S_.lu().solve(C);
    P = Pbar - Pbar * C.transpose() * SC * Pbar;
    P = 0.5 * (P + P.transpose());

    // // reduce position drift
    if (P.block<2, 2>(0, 0).determinant() > 1e-6) {
        P.block<2, 16>(0, 2).setZero();
        P.block<16, 2>(2, 0).setZero();
        P.block<2, 2>(0, 0) /= 10.0;
    }
 
    // final step
    // put estimated values back to A1CtrlStates& state
    // for (int i = 0; i < NUM_LEG; ++i) {
    //     if (estimated_contacts[i] < 0.5) {
    //        x_estimated_contacts[i] = false;
    //     } else {
    //         x_estimated_contacts[i] = true;
    //     }
    // }
    // x_estimated_root_pos = x.segment<3>(0);
    // x_estimated_root_vel = x.segment<3>(3);

    // x_root_pos = x.segment<3>(0);
    // x_root_lin_vel = x.segment<3>(3);
    for (int i = 0; i < 3; i++)
    {
       estimated_root_pos[i] = x(0 + i);  //机身质心全局位置
       estimated_root_vel[i] =  x(3 + i);  //机身质心全局速度
    }
     root_pos = x.segment<3>(0).transpose();
     root_lin_vel = x.segment<3>(3).transpose();
     for (int i = 0; i < 4; i++)
     {
       Touch_footposition.col(i) = x.segment<3>(6 + i * 3);
     }
    std::cout << "估计的一号腿的位置："<< Touch_footposition.col(0) << std::endl;
    std::cout << "估计的二号腿的位置："<< Touch_footposition.col(1) << std::endl;
}

void EKF::dataBusRead(DataBus &robotState){
       
        root_rot_mat = robotState.base_rot;

        imu_acc(0,0) = robotState.baseAcc[0];  //加速度
        imu_acc(1,0) = robotState.baseAcc[1];
        imu_acc(2,0) = robotState.baseAcc[2];

        x(3,0) = robotState.baseLinVel[0];  //速度
        x(4,0) = robotState.baseLinVel[1];  
        x(5,0) = robotState.baseLinVel[2];  
        std::cout << x(3,0) <<"  "<< x(4,0) <<"   "<< x(5,0) <<std::endl;

        // std::cout << "观测出来的IMU加速度：" << imu_acc.transpose() << std::endl;

        foot_pos_rel.block<3,1>(0,0) = robotState.fe_l_pos_L;  //left 
        foot_pos_rel.block<3,1>(0,1) = robotState.fe_r_pos_L;  //right
        foot_pos_rel.block<3,1>(0,2) = robotState.fe_l_pos_L;  //left
        foot_pos_rel.block<3,1>(0,3) = robotState.fe_r_pos_L;  //right

        // std::cout << " foot_pos_rel.block<3,1>(0,0)" << foot_pos_rel.block<3,1>(0,0) << std::endl;
        
        // 机身坐标系下的角速度
        imu_ang_vel = robotState.base_omega_W; 

        // 机身坐标系下的足端速度
        foot_vel_rel.block<3,1>(0,0) = robotState.fe_l_vel_L;
        foot_vel_rel.block<3,1>(0,1) = robotState.fe_r_vel_L;
        foot_vel_rel.block<3,1>(0,2) = robotState.fe_l_vel_L;
        foot_vel_rel.block<3,1>(0,3) = robotState.fe_r_vel_L;

        if(robotState.motionState == DataBus::Stand)
        {
             movement_mode = 0;
        }
        else
        {
             movement_mode = 1;
        }
  
        foot_force(0, 0) = robotState.FL_est[2]; //left leg
        foot_force(1, 0) = robotState.FR_est[2]; //right leg
        foot_force(2, 0) = robotState.FL_est[2]; //left leg
        foot_force(3, 0) = robotState.FR_est[2]; //right leg
        std::cout << "估计出来的足端力：" << foot_force.transpose() << std::endl;

};


void EKF::dataBusWrite(DataBus &robotState){

   robotState.basePos[0] = x(0,0);
   robotState.basePos[1] = x(1,0);
   robotState.basePos[2] = x(2,0)+0.07;
   robotState.fe_l_pos_W = Touch_footposition.col(0);
   robotState.fe_r_pos_W = Touch_footposition.col(1);
   std::cout << "估计出来的质心位置: " << robotState.basePos[0] << " " << robotState.basePos[1] << " " << robotState.basePos[2] << std::endl;

};
void EKF::update_dt(double dtin)
{
    dt = dtin;
}

Eigen::Matrix3d skew(Eigen::Vector3d vec) {
    Eigen::Matrix3d rst; rst.setZero();
    rst <<            0, -vec(2),  vec(1),
            vec(2),             0, -vec(0),
            -vec(1),  vec(0),             0;
    return rst;
}











