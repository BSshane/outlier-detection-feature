# 工地异常检测系统

## 项目简介
本项目是一个综合性的工地异常检测系统，整合了人脸识别、行为分析、声音监控和工地安全检测等多种功能，旨在通过计算机视觉和音频分析技术实现对施工现场人员行为和环境的实时监控、智能预警和数据分析，从而显著提升工地安全管理水平。

## 主要功能
- **人脸识别**：基于Dlib的实时人脸检测与识别
- **行为分析**：使用OpenPose进行人体姿态估计和异常行为识别
- **声音监控**：检测异常声音（如尖叫声）并触发报警
- **工地安全**：基于YOLOv5的危险区域入侵检测和安全装备识别
- **Web监控界面**：提供实时监控和报警信息展示的Web界面

## 项目结构
```
智能安防系统
├── 核心算法模块
│   ├── Dlib_face_recognition_from_camera/    # 人脸识别系统
│   │   ├── get_faces_from_camera.py          # 人脸注册
│   │   ├── face_reco_from_camera.py          # 实时识别
│   │   └── data/                             # 人脸特征库
│   │
│   ├── Smart_Construction-master/            # 施工安全检测
│   │   ├── detect.py                         # YOLOv5检测入口
│   │   ├── area_detect.py                    # 危险区域分析
│   │   └── utils/                            # 数据预处理工具
│   │
│   ├── Sound_monitor/                        # 声音检测
│   │   ├── backend.py                        # 服务启动入口
│   │   ├── video_test.py                     # 视频声音检测测试
│   │   └── model/                            # YAMNet模型
│   │
│   └── ism_person_openpose-master/           # 姿态分析
│       ├── runOpenpose.py                    # OpenPose入口
│       └── openpose_modules/                 # 关键点检测算法
│
├── 系统服务
│   ├── GetStream/                            # 视频流处理&前端服务
│   │   ├── start.ps1                         # PowerShell启动脚本
│   │   └── debug_algorithm_b.ps1             # 算法调试
│   │
│   └── ServerPage/                           # 数据库操作&鉴权服务
│       ├── backend.py                        # 服务接口
│       └── frontend/                         # Vue3前端
│
└── 辅助系统
    ├── Daily/                                # 安全日报系统
    │   ├── DailyReport/                      # 日报存档
    │   └── cot_prompt.py                     # 报告生成AI提示
    │
    └── FileGetter/                           # 本地文件管理(用于告警中心)
        └── start.bat                         # 服务启动脚本
```

## 服务启动

1.  ServerPage
    1.  在服务端安装Docker
    2.  在服务端执行start.sh
2.  ServerPage的Jenkins自动部署
    1.  在服务器安装Jenkins
    2.  将Github/Gitee账号导入Jenkins
    3.  在仓库设置Webhook
    4.  在Jenkins中配置Webhook
    5.  配置Jenkins的构建触发器(修改master/develop/feature时触发)
    6.  配置Jenkins的构造步骤(执行start.sh)
3.  GetStream
4.  FileGetter
    1.  运行start.bat
5.  Daily
    1.  按照environment.yml配置好环境
    2.  运行cot_prompt.py
6.  Sound_monitor
    1.  在电脑上安装ffmpeg
    2.  按照environment.yml配置好环境
    3.  运行backend.py
7.  Dlib_face_recognition_from_camera
    1.  按照env.yml配置好环境
    2.  运行feature_GetAndConvert.py录入人脸信息.
    3.  运行app_cnn_..._.py(文件名最长的那个)启动服务.
8.  Smart_Construction-master
9.  ism_person_openpose-master