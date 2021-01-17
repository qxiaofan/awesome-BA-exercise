#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <fstream>

using namespace Eigen;
using namespace ceres;
using namespace std;

class ReprojectionError {
public:

    ReprojectionError(Eigen::Vector3d point_, Eigen::Vector2d observed_)
            : point(point_), observed(observed_) {
    }

    template<typename T>
    bool operator()(const T *const camera_r, const T *const camera_t, T *residuals) const {
        T pt1[3];
        pt1[0] = T(point.x());
        pt1[1] = T(point.y());
        pt1[2] = T(point.z());

        T pt2[3];
        ceres::AngleAxisRotatePoint(camera_r, pt1, pt2);

        pt2[0] = pt2[0] + camera_t[0];
        pt2[1] = pt2[1] + camera_t[1];
        pt2[2] = pt2[2] + camera_t[2];

        const T xp = T(K[0] * (pt2[0] / pt2[2]) + K[2]);
        const T yp = T(K[1] * (pt2[1] / pt2[2]) + K[3]);

        const T u = T(observed.x());
        const T v = T(observed.y());

        residuals[0] = u - xp;
        residuals[1] = v - yp;

        LOG(INFO) << residuals[0] << " " << residuals[1];

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Vector3d points, Eigen::Vector2d observed) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
                new ReprojectionError(points, observed)));
    }

private:
    Eigen::Vector3d point;
    Eigen::Vector2d observed;
    // Camera intrinsics
    double K[4] = {520.9, 521.0, 325.1, 249.7}; // fx,fy,cx,cy
};

vector<Eigen::Vector3d> p3d;
vector<Eigen::Vector2d> p2d;
typedef Eigen::Matrix3d rotation_t;
typedef Eigen::Vector3d rodrigues_t;

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

bool readData() {
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
            p3d.push_back(Point3d);
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
            p2d.push_back(Point2d);
        }
    }
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    cout << "Read " << nPoints << " points." << endl;
    return true;
}

int main(int argc, char *argv[]) {

    // Google log
    google::InitGoogleLogging(argv[0]);
//    google::SetLogDestination(google::GLOG_INFO, "../../log/ceres_log_");

    if (!readData()) {
        cerr << "Read data failed." << endl;
        return -1;
    }

//    Eigen::Matrix3d K;
//    K << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;

    ceres::Problem problem;
    ceres::LossFunction *lossfunction = NULL;
    double camera_rvec[3] = {0, 0, 0};
    double camera_t[3] = {0, 0, 0};

    for (uint i = 0; i < p3d.size(); i++) {
        Eigen::Vector3d p3dVec(p3d[i](0), p3d[i](1), p3d[i](2));
        Eigen::Vector2d p2dVec(p2d[i](0), p2d[i](1));
        ceres::CostFunction *costfunction = ReprojectionError::Create(p3dVec, p2dVec);
        problem.AddResidualBlock(costfunction, lossfunction, camera_rvec, camera_t);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 100;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    Eigen::Vector3d omega;
    omega << camera_rvec[0], camera_rvec[1], camera_rvec[2];
    rotation_t rMatrix = rodrigues2rot(omega);

    Eigen::Matrix4d T;
    T << rMatrix(0, 0), rMatrix(0, 1), rMatrix(0, 2), camera_t[0],
            rMatrix(1, 0), rMatrix(1, 1), rMatrix(1, 2), camera_t[1],
            rMatrix(2, 0), rMatrix(2, 1), rMatrix(2, 2), camera_t[2],
            0, 0, 0, 1;
    std::cout << "T = \n" << T << std::endl;
    return 0;
}