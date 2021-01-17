# BA_exercise
### 前言

本仓库forked from [BA_exercise](https://github.com/shanpenghui/BA_exercise)，感谢我们[3D视觉从入门到精通](https://mp.weixin.qq.com/s/weShDMbGTf0amg1qu_t8cw)星球嘉宾鹏辉的分享与整理。

本demo对应的理论部分，见[Bundle Adjustment原理及应用](https://mp.weixin.qq.com/s/hIxM3dNCmL6kb3_VW_66-g)



### 正文

A exercise of BA, using g2o, ceres and eigen
Before build the source, make sure that the third party library is installed rightly.
其中g2o版本是六哥的版本，感谢六哥。同时也感谢黄志明的大力支持和解惑，非常感谢。

# ThirdParty
>googlelog
>
>g2o
>
>ceres

# Clone
git clone --recurse-submodules -j8 https://github.com/shanpenghui/BA_exercise

# Usage

## 1.Build g2o
```
cd Thirdparty/g2o
mkdir build
cd build
cmake ..
make -j4
```

## 2.Build ceres
```
cd Thirdparty/ceres-solver
mkdir build
cd build
cmake ..
make -j4
```

## 3.Start BA_g2o
```
cd g2o
mkdir build
cd build
cmake ..
make -j4
./BA_g2o
```

## 4.Start BA_ceres
>Before running BA_ceres, you can get used to using ceres by the examples offered in source folder named ceres
```
cd ceres
mkdir build
cd build
cmake ..
make -j4
./BA_ceres
```

## 5.Start BA_eigen
```
cd eigen
mkdir build
cd build
cmake ..
make -j4
./BA_eigen
```

# 输出
## g2o
![image](https://github.com/shanpenghui/BA_exercise/blob/main/imgs/g2o.png)
## ceres
![image](https://github.com/shanpenghui/BA_exercise/blob/main/imgs/ceres.png)
## eigen
![image](https://github.com/shanpenghui/BA_exercise/blob/main/imgs/eigen.png)

# 备注

当前版本是没有优化point pose的，即只优化位姿

