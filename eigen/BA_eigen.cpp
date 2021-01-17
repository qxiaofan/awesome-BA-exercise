//
// Created by sph on 2021/1/12.
//

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>

using namespace std;
using namespace Eigen;

// 定义位姿数据结构
typedef Eigen::Matrix3d rotation_t;
typedef Eigen::Vector3d rodrigues_t;
typedef Eigen::Vector3d Point3D;
typedef Eigen::Vector2d Point2D;
typedef Eigen::Matrix4d SE3;
struct se3 {
    rodrigues_t r_;
    rodrigues_t t_;
};
struct Edge {
    int xi, xj;
    Eigen::Vector3d measurement;
    Eigen::Matrix3d infoMatrix;
};
struct Graph {
    std::vector<std::pair<SE3, std::vector<Point2D> > > graph_;
};

// 相机内参
double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
bool debug_ = false;

/**
 * @brief 旋转向量转换成旋转矩阵，即李群->李代数
 * @param[in]  omega        旋转向量
 * @param[out] R            旋转矩阵
 */
Eigen::Matrix3d rodrigues2rot(const Eigen::Vector3d &omega) {
    // 初始化旋转矩阵
    rotation_t R = Eigen::Matrix3d::Identity();

    // 求旋转向量的反对称矩阵
    Eigen::Matrix3d skewW;
    skewW << 0.0, -omega(2), omega(1),
            omega(2), 0.0, -omega(0),
            -omega(1), omega(0), 0.0;

    // 求旋转向量的角度
    double omega_norm = omega.norm();

    // 通过罗德里格斯公式把旋转向量转换成旋转矩阵
    if (omega_norm > std::numeric_limits<double>::epsilon())
        R = R + std::sin(omega_norm) / omega_norm * skewW
            + (1 - std::cos(omega_norm)) / (omega_norm * omega_norm) * (skewW * skewW);

    return R;
}

/**
 * @brief 旋转矩阵转换成旋转向量，即李代数->李群
 * @param[in]  R       旋转矩阵
 * @param[out] omega   旋转向量
 问题: 李群(SO3) -> 李代数(so3)的对数映射
 利用罗德里格斯公式，已知旋转矩阵的情况下，求解其对数映射ln(R)
 就像《视觉十四讲》里面说到的，使用李代数的一大动机是为了进行优化,而在优化过程中导数是非常必要的信息
 李群SO(3)中完成两个矩阵乘法，不能用李代数so(3)中的加法表示，问题描述为两个李代数对数映射乘积
 而两个李代数对数映射乘积的完整形式，可以用BCH公式表达，其中式子(左乘BCH近似雅可比J)可近似表达，该式也就是罗德里格斯公式
 计算完罗德里格斯参数之后，就可以用非线性优化方法了，里面就要用到导数，其形式就是李代数指数映射
 所以要在调用非线性MLPnP算法之前计算罗德里格斯参数
 */
Eigen::Vector3d rot2rodrigues(const Eigen::Matrix3d &R) {
    rodrigues_t omega;
    omega << 0.0, 0.0, 0.0;

    // R.trace() 矩阵的迹，即该矩阵的特征值总和
    double trace = R.trace() - 1.0;

    // 李群(SO3) -> 李代数(so3)的对数映射
    // 对数映射ln(R)∨将一个旋转矩阵转换为一个李代数，但由于对数的泰勒展开不是很优雅所以用本式
    // wnorm是求解出来的角度
    double wnorm = std::acos(trace / 2.0);

    // 如果wnorm大于运行编译程序的计算机所能识别的最小非零浮点数，则可以生成向量，否则为0
    if (wnorm > std::numeric_limits<double>::epsilon()) {
        //        |r11 r21 r31|
        //   R  = |r12 r22 r32|
        //        |r13 r23 r33|
        omega[0] = (R(2, 1) - R(1, 2));
        omega[1] = (R(0, 2) - R(2, 0));
        omega[2] = (R(1, 0) - R(0, 1));
        //             theta       |r23 - r32|
        // ln(R) =  ------------ * |r31 - r13|
        //          2*sin(theta)   |r12 - r21|
        double sc = wnorm / (2.0 * std::sin(wnorm));
        omega *= sc;
    }
    return omega;
}

bool readData(std::vector<Point3D> &VertexPoints3D, std::vector<Point2D> &VertexPoints2D) {
    std::string p3d_file = "../../data/p3d.txt";
    std::string p2d_file = "../../data/p2d.txt";
    // 导入3D点和对应的2D点
    ifstream fp3d(p3d_file);
    if (!fp3d) {
        cout << "No p3d.text file" << endl;
        return false;
    } else {
        while (!fp3d.eof()) {
            double pt3[3] = {0};
            for (auto &p:pt3) {
                fp3d >> p;
            }
            Eigen::Vector3d Point3d;
            Point3d << pt3[0], pt3[1], pt3[2];
            VertexPoints3D.push_back(Point3d);
        }
    }
    ifstream fp2d(p2d_file);
    if (!fp2d) {
        cout << "No p2d.text file" << endl;
        return false;
    } else {
        while (!fp2d.eof()) {
            double pt2[2] = {0};
            for (auto &p:pt2) {
                fp2d >> p;
            }
            Eigen::Vector2d Point2d;
            Point2d << pt2[0], pt2[1];
            VertexPoints2D.push_back(Point2d);
        }
    }
    assert(VertexPoints3D.size() == VertexPoints2D.size());

    int nPoints = VertexPoints3D.size();
    cout << "Read " << nPoints << " points." << endl;
    return true;
}

void ComputeJacobian(Eigen::MatrixXd &JacobianPose, std::vector<Point3D> &points, std::vector<SE3> &VertexPoses) {
    // 待优化的位姿节点数量
    int N_poses = VertexPoses.size();
    // 3D坐标点的节点数量
    int N_points = points.size();
    //                                                Pose1            Pose2                PoseN_poses
    //                     /     Point1     |        J1(2×6)                                                 |
    //              Pose1         ...       |         ...              0(2×6)       ...        0(2×6)        |
    //                     \ PointN_points  |     JN_points(2×6)                                             |
    //                     /     Point1     |                          J1(2×6)                               |
    //              Pose2         ...       |        0(2×6)             ...         ...        0(2×6)        |
    //                     \ PointN_points  |                       JN_points(2×6)                           |
    // JacobianPose =              ⋮        |          ⋮                ⋮                       ⋮            |
    //                     /     Point1     |                                                 J1(2×6)        |
    //        PoseN_poses         ...       |        0(2×6)            0(2×6)       ...        ...           | (2·N_points × 6)·N_poses =
    //                     \ PointN_points  |                                              JN_points(2×6)    | (2·N_poses·N_points × 6·N_poses)
    JacobianPose = Eigen::MatrixXd::Zero(2 * N_poses * N_points, 6 * N_poses);
    if(debug_) {
        cout << "JacobianPose : rows(" << JacobianPose.rows() <<  ") cols(" << JacobianPose.cols() << ")" << endl;
    }

    // 循环用的Jacobian，放在循环外部初始化
    Eigen::MatrixXd JacobianPose_i_(Eigen::MatrixXd::Zero(2, 6));
    // 这里是假设所有的位姿都可以看到所有的3D点，所以才可以循环重复遍历，否则如果有些位姿看不到某些3D点，需要对3D点进行标记并选取位姿看得到的3D点进行Jaco矩阵求导
    // 这个例子其实就只优化了一个位姿，所以该位姿当然可以看到所有的3D点，外部循环只是想做通用，但是这里效果不大
    for (int j = 0; j < N_poses; ++j) {
        for (int i = 0; i < N_points; ++i) {

            // P' = R*P + t
            Eigen::MatrixXd R_ = VertexPoses[j].block<3, 3>(0, 0);
            Eigen::MatrixXd t_ = VertexPoses[j].block<3, 1>(0, 3);
            Eigen::Vector3d P_ = R_ * points[i] + t_;
            double x = P_(0, 0), y = P_(1, 0), z = P_(2, 0), z_2 = z * z;

            // Jacobian of pose : のf/のT
            // 只优化位姿，所以只有位姿导数，没有坐标点导数
            JacobianPose_i_(0, 0) = -1. / z * fx;
            JacobianPose_i_(0, 1) = 0;
            JacobianPose_i_(0, 2) = x / z_2 * fx;
            JacobianPose_i_(0, 3) = x * y / z_2 * fx;
            JacobianPose_i_(0, 4) = -(1 + (x * x / z_2)) * fx;
            JacobianPose_i_(0, 5) = y / z * fx;
            JacobianPose_i_(1, 0) = 0;
            JacobianPose_i_(1, 1) = -1. / z * fy;
            JacobianPose_i_(1, 2) = y / z_2 * fy;
            JacobianPose_i_(1, 3) = (1 + (y * y / z_2)) * fy;
            JacobianPose_i_(1, 4) = -x * y / z_2 * fy;
            JacobianPose_i_(1, 5) = -x / z * fy;

            JacobianPose.block<2, 6>(2 * j * N_points + 2*i, 6 * j) = JacobianPose_i_;

        }
    }

    if(debug_) {
        cout << JacobianPose << endl;
    }
}

double ComputeReprojectionError(Eigen::VectorXd &ReprojectionError, std::vector<Point3D> &points3d,
                                std::vector<SE3> &VertexPoses, std::vector<Point2D> &Points2D) {
    // 待优化的位姿节点数量
    int N_poses = VertexPoses.size();
    // 3D坐标点的节点数量
    int N_points3d = points3d.size();
    double reprojection_error = 0.;
    //                       /     Point1     |       error1(2×1)      |
    //                 Pose1        ...       |        ...             |
    //                       \ PointN_points  |   errorN_points(2×1)   |
    // ReprojectionError =           ⋮        |         ⋮              |
    //                       /     Point1     |       error1(2×1)      |
    //           PoseN_poses        ...       |        ...             |
    //                       \ PointN_points  |   errorN_points(2×1)   | (2·N_points·N_poses × 1)

    // 要和 JacobianPose 转置相乘，而 JacobianPose^T (6·N_poses × 2·N_poses·N_points)
    ReprojectionError = Eigen::VectorXd::Zero(2 * N_poses * N_points3d, 1);
    if(debug_) {
        cout << "ReprojectionError : rows(" << ReprojectionError.rows() <<  ") cols(" << ReprojectionError.cols() << ")" << endl;
    }
    for (int j = 0; j < N_poses; ++j) {
        for (int i = 0; i < N_points3d; ++i) {
            Eigen::MatrixXd JacobianPose_i_(Eigen::MatrixXd::Zero(2, 6));
            // P' = R*P + t
            Eigen::MatrixXd R_ = VertexPoses[j].block<3, 3>(0, 0);
            Eigen::MatrixXd t_ = VertexPoses[j].block<3, 1>(0, 3);
            if(debug_) {
//                cout << "R_ : " << R_ << endl;
//                cout << "t_ : " << t_ << endl;
            }
            Eigen::Vector3d P_ = R_ * points3d[i] + t_;
            double x = P_(0, 0), y = P_(1, 0), z = P_(2, 0), z_2 = z * z;
            // 观测值
            double p_u = cx + fx * x / z, p_v = cy + fy * y / z;
            // 误差项
            double du = Points2D[i](0, 0) - p_u, dv = Points2D[i](1, 0) - p_v;
            ReprojectionError(j * N_points3d * 2 + 2 * i, 0) = du;
            ReprojectionError(j * N_points3d * 2 + 2 * i + 1, 0) = dv;
            reprojection_error += (du * du + dv * dv);
        }
    }
    if(debug_) {
        cout << "reprojection error : " << reprojection_error / (N_poses * N_points3d) << endl;
//        cout << ReprojectionError << endl;
    }
    return reprojection_error / (N_poses * N_points3d);
}

bool isFinished(const double current_error, const double previous_error) {
    if (previous_error < 0.0) {
        // This case is 1st optimization
        // do nothing
        return false;
    }
    if (previous_error <= current_error + 1e-5) {
        return true;
    }
}

// 计算李代数的增量
Eigen::MatrixXd ComputeSe3(const Eigen::MatrixXd &JacobianPose, const Eigen::VectorXd &ReprojectionError) {
    Eigen::MatrixXd H = JacobianPose.transpose() * JacobianPose;
    if(debug_) {
//        cout << "H : rows(" << H.rows() <<  ") cols(" << H.cols() << ")" << endl;
    }

    bool chose_ldlt = true;
    if(chose_ldlt) {
        Eigen::MatrixXd g = -JacobianPose.transpose() * ReprojectionError;
        Eigen::MatrixXd delta_x_ldlt = H.ldlt().solve(g);
        if(debug_) {
//            cout << "delta_x_ldlt : rows(" << delta_x_ldlt.rows() <<  ") cols(" << delta_x_ldlt.cols() << ")" << endl;
//            cout << delta_x_ldlt << endl;
        }
        return delta_x_ldlt;
    }

    Eigen::MatrixXd minus_b = -1.0 * ReprojectionError;
    Eigen::MatrixXd delta_x = H.inverse() * JacobianPose.transpose() * minus_b;

    if(debug_) {
        cout << "minus_b : rows(" << minus_b.rows() <<  ") cols(" << minus_b.cols() << ")" << endl;
        cout << "delta_x : rows(" << delta_x.rows() <<  ") cols(" << delta_x.cols() << ")" << endl;
        cout << delta_x << endl;
    }
    return delta_x;
}

// 利用李代数的增量更新李群
void UpdateSE3(std::vector<SE3> &VertexPoses, Eigen::VectorXd delta_se3) {
    // 待优化的位姿节点数量
    int N_poses = VertexPoses.size();
    for (int i = 0; i < N_poses; ++i) {
        // get delta_se3
        se3 delta_se3_i;
        delta_se3_i.r_ = delta_se3.block<3,1>(3,0);
        delta_se3_i.t_ = delta_se3.block<3,1>(0,0);

        if(debug_) {
//            cout << "delta_se3_i.r_ : rows(" << delta_se3_i.r_.rows() <<  ") cols(" << delta_se3_i.r_.cols() << ")" << endl;
//            cout << delta_se3_i.r_ << endl;
//            cout << "delta_se3_i.t_ : rows(" << delta_se3_i.t_.rows() <<  ") cols(" << delta_se3_i.t_.cols() << ")" << endl;
//            cout << delta_se3_i.t_ << endl;
        }

        // map se3 to SE3
        Eigen::Matrix3d rot_ = rodrigues2rot(delta_se3_i.r_);
        if(debug_) {
//            cout << "rot_ : rows(" << rot_.rows() << ") cols(" << rot_.cols() << ")"
//                 << endl;
//            cout << rot_ << endl;
        }

        // create SE3 for delta_se3
        SE3 delta_SE3_i;
        delta_SE3_i.block<3,3>(0,0) = rot_;
        delta_SE3_i.block<3,1>(0,3) = delta_se3_i.t_;
        delta_SE3_i(3,3) = 1;
        if(debug_) {
//            cout << "delta_SE3_i : rows(" << delta_SE3_i.rows() << ") cols(" << delta_SE3_i.cols() << ")"
//                 << endl;
//            cout << delta_SE3_i << endl;
        }

        // Update pose_i
        VertexPoses[i] = delta_SE3_i * VertexPoses[i];

        if(debug_) {
//            cout << "VertexPoses[" << i << "] : rows(" << VertexPoses[i].rows() <<  ") cols(" << VertexPoses[i].cols() << ")" << endl;
//            cout << VertexPoses[i] << endl;
        }
    }

}

void Gaussian_Newton(const int iterations, std::vector<SE3> &poses, vector<Point3D> &p3ds,
                     vector<Point2D> &p2ds) {

    // 初始化数据
    Eigen::MatrixXd J_T;
    Eigen::VectorXd Error;
    double cur_error, pre_error = -1;

    // 循环求解
    for (int i = 0; i < iterations; ++i) {
        // compute Jacobian
        ComputeJacobian(J_T, p3ds, poses);
        // compute error
        cur_error = ComputeReprojectionError(Error, p3ds, poses, p2ds);
        // if error become more greater, stop
//        if(isFinished(cur_error, pre_error)) {
//            cout << "Iteration finished, stop." << endl;
//            return;
//        }
//        pre_error = cur_error;
        // computer delta_x
        Eigen::VectorXd delta_se3 = ComputeSe3(J_T, Error);
        // 利用delta的值来停止迭代效果更好
        if(delta_se3.norm() < 1e-5) break;
        if (isnan(delta_se3(0,0))) {
            cout << "delta_se3 is nan!" << endl;
            break;
        }
        UpdateSE3(poses, delta_se3);
    }

}

int main(int argc, char **argv) {

    // step 1: 读取3D和2D点数据
    std::vector<Point3D> VertexPoints3D;    // 3D点为节点
    std::vector<Point2D> Points2D;          // 2D点不是节点，而是作为观测值
    if (!readData(VertexPoints3D, Points2D)) {
        cerr << "Read data failed." << endl;
        return -1;
    }

    // step 2: 非线性优化

    // 初始化位姿
    std::vector<SE3> VertexPoses;
    SE3 initial_pose(Eigen::Matrix4d::Identity());
    VertexPoses.push_back(initial_pose);

    // 高斯牛顿解非线性方程
    Gaussian_Newton(100, VertexPoses, VertexPoints3D, Points2D);
    for (int i = 0; i < VertexPoses.size(); ++i) {
        cout << "VertexPoses[" << i << "] : rows(" << VertexPoses[i].rows() <<  ") cols(" << VertexPoses[i].cols() << ")" << endl;
        cout << VertexPoses[i] << endl;
    }
    return 0;
}


