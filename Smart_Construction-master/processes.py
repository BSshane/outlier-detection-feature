# processes.py
import time
import logging
import pymysql
from config import risk_queue, stop_event, db_config
import json

# 配置日志
def setup_process_logger(name):
    """配置独立的进程日志"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"{name}.log")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger

def reader_process(risk_pool):
    """风险队列读者进程 - 从队列中取出风险信息并写入数据库"""
    logger = setup_process_logger("RiskReaderProcess")
    logger.info("读者进程已启动，等待风险信息...")

    # 连接数据库
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        logger.info("成功连接到数据库")
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        return

    while not stop_event.is_set():
        try:
            # 从队列获取风险信息
            risk_info = risk_queue.get(timeout=1)
            logger.info(f"从队列获取风险信息: {risk_info.get('warning', '未知风险')}")

            # 打印风险信息详情
            logger.debug(f"风险信息详情: {risk_info}")

            # 写入数据库
            try:
                sql = """
                      INSERT INTO warning (cameraId, type, videoURL, info)
                      VALUES (%s, %s, %s, %s) \
                      """
                cursor.execute(sql, (
                    risk_info['cameraId'],
                    risk_info['type'],
                    risk_info['videoURL'],
                    json.dumps(risk_info['info'])
                ))
                conn.commit()
                logger.info(f"风险信息已写入数据库: {risk_info.get('warning', '未知风险')}")
                risk_pool.append(risk_info)
            except pymysql.Error as e:
                # 记录数据库错误详情
                logger.error(f"数据库操作失败: {str(e)}", exc_info=True)
                conn.rollback()
                # 检查数据库连接是否有效
                if not conn.open:
                    logger.warning("数据库连接已关闭，尝试重连...")
                    conn = pymysql.connect(**db_config)
                    cursor = conn.cursor()

        except Exception as e:
            # 捕获所有异常并记录完整堆栈信息
            logger.error(f"处理风险队列时出错: {str(e)}", exc_info=True)
            # 避免因单个错误导致进程崩溃
            continue

    # 清理资源
    cursor.close()
    conn.close()
    logger.info("读者进程已停止")