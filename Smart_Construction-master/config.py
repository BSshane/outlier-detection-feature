# config.py
import pymysql
from multiprocessing import Queue, Event
from pathlib import Path
import multiprocessing
# 视频流配置
rtmp_url = 'rtmp://1.92.135.70:9090/live/1'

# 数据库配置
db_config = {
    'host': '1.92.135.70',
    'port': 3306,
    'user': 'root',
    'password': 'Aa123321',
    'database': 'AbnormalDetection',
}

# 风险池配置
MAX_RISK_POOL_SIZE = 100

# 进程间通信组件（全局唯一）
risk_queue = multiprocessing.Queue()  # 风险信息队列
stop_event = Event()              # 进程终止事件

# 危险区域（初始值）
current_danger_zone = (100, 100, 700, 700)

# 查找权重文件
def get_weights_path():
    script_dir = Path(__file__).parent.resolve()
    weight_root = script_dir.joinpath("weights")
    if not weight_root.exists():
        raise FileNotFoundError(f"未找到权重文件夹: {weight_root}")
    weight_files = [item for item in weight_root.iterdir() if item.suffix == ".pt"]
    if not weight_files:
        raise FileNotFoundError(f"在 {weight_root} 中未找到 .pt 权重文件")
    return str(weight_files[0])