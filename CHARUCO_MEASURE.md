# 任务2：双标定板实时测量

运行（在项目目录）：

```bash
python3 measure_charuco.py
python3 measure_charuco.py --headless --jsonl measurements.jsonl
```

默认连接可用的RealSense相机，只启用1280×720、30FPS的彩色流，并从该实际启用的视频流读取设备内参。双板PnP不请求深度、加速度或陀螺仪，避免这些额外流导致Couldn't resolve requests。多相机时用--serial指定。启动时打印序列号、分辨率、焦距、主点及畸变。

仅读取内参（不进入测量循环）：

```bash
python3 measure_charuco.py --read-intrinsics
python3 measure_charuco.py --read-intrinsics --save-intrinsics config/device_intrinsics.yaml
```

测量时保存本次内参：`python3 measure_charuco.py --save-intrinsics config/device_intrinsics.yaml`。读取失败会明确报错，不自动套用旧文件。需要原有文件模式时使用--intrinsics-source file，并用--serial指定匹配设备。非零逆Brown等不支持的畸变模型可以读取保存，但会阻止直接用于PnP。
需要 numpy、PyYAML、带aruco的OpenCV和pyrealsense2。USB相机用 `--source 0 --intrinsics 对应内参.yaml`；视频也可以传给--source。不同相机必须提供自己的内参。
窗口同时绘制两块板的角点与坐标轴，并分别显示Board1（顶块）、Board2（小车）的识别状态、有效角点数和重投影误差。即使1号板丢失，也显示2号板状态；JSON逐帧包含board1、board2及board2_mount_configured。缺少R_car_board2或t_car_board2时明确提示双板车坐标不可用。当前采集仅启用彩色流，不提供重力兜底。

窗口按q或Esc退出；无窗口模式按Ctrl+C。窗口与终端用中文显示X、Y、Z、yaw，终端每秒约5次输出；--jsonl仍保存每帧数值JSON。yaw以1号纸下方为顶块方向基准，车前为0°、向车右为正；optical_yaw_deg是该方向在相机XZ平面的投影，car_yaw_deg是该方向在车身XY平面的投影，投影不足时为null。

输出：顶块200mm立方体几何中心在彩色相机系的XYZ（右、下、前）、到彩色光心的直线距离；在小车右前角系的XYZ（右、前、上）、三维直线距离与XY水平距离。JSON长度单位米，窗口毫米。两块板应在同一帧可见。目标板丢失立即停止输出坐标，不保持旧值。

## 真实参数与待核实项

项目已有的安装常量在cube/cap_pose.py：光心相对车右前角为(-0.211,0.050,0.669)m。其文字注释却写“右侧”，符号冲突。新程序将实际常量单独写入config/charuco_measure.yaml，未确认时标记PROVISIONAL。

配置默认1号板ID0–23、2号板ID40–63，8×6格，格长30mm，标记22mm；应尺量实际打印格长，不能只信纸上文字。当前配置按现代OpenCV默认棋盘布局；若实物检测不到，需核对生成文件的legacyPattern和ID排列。

棋盘坐标：文字正向看，完整棋盘外框左上角为原点，X向右、Y向下，Z进入板背面。棋盘中心为(120,90,0)mm。

已确认待测顶块边长200mm，最近顶角位于1号纸中心的左上方。纸中心到该顶角为(-125,-168.5,+3)mm；顶角到体心为(+100,+100,+100)mm。因此体心相对纸中心为(-25,-68.5,103)mm。纸的印刷面比顶块顶面高3mm，体心在纸背面103mm。350mm白盒不参与200mm顶块的体心偏移计算。

按用户要求忽略棋盘中心与A4纸中心的2mm偏移，board_center_to_a4_center为零；cube_mount_confirmed为true。这是基于确认的安装几何，仍需实机量测验证打印尺度和误差。

2号车上板的纸上边朝摄像头（车前），已按确认的180°反向安装修正。图1为俯视图，CAD的X向图下、Y向图右，车前向图左；程序车系固定为X向车右、Y向车前、Z向上，所以程序车系=(-CAD_X,-CAD_Y,CAD_Z)。图中从车右前角到纸中心的CAD位移(175,134.8,570)mm，转换到程序车系为(-175,-134.8,570)mm。

2号板X向车右、Y向车后、Z向车下，故R_car_board2=diag(1,-1,-1)。棋盘中心在板系(120,90,0)mm，棋盘左上角位置t_car_board2=纸中心-R_car_board2×棋盘中心=(-295,-44.8,570)mm。配置已启用此变换并设置board2_mount_confirmed=true。同帧识别两块板后优先用双板计算，不依赖相机安装位置或重力。

JSON的optical_axes为[right,down,forward]，car_axes为[right,forward,up]；坐标和距离单位均为米，窗口显示毫米。

当前程序只启用彩色流，不提供实时重力；2号板丢失或缺少安装变换时，不输出小车坐标，1号板仍可提供光心坐标与距离。measure函数保留对外传入有效重力的接口，但该备用路径的相机安装常量尚未确认，不作为本次双板测量依据。

本环境尚未完成相机实机验证。运行前确认摄像头由运行程序的主机实际连接、没有被其他程序占用。
