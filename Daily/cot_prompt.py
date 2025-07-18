import pandas as pd
from datetime import datetime
import os
import traceback
from openai import OpenAI
import pymysql
from flask import Flask, request, jsonify  # 新增Flask导入
from flask_restx import Api, Resource, fields
from flask_cors import CORS  # 添加CORS导入


# 千帆配置类
class QianfanConfig:
    def __init__(self):
        self.API_KEY = "bce-v3/ALTAK-Xw2BdfqQNXMXQJcHqpoLM/dd269dc88961bc4ab35764116355d6785641e7e0"
        self.BASE_URL = "https://qianfan.baidubce.com/v2"
        self.MODEL = "ernie-4.0-turbo-8k"

    def get_client(self):
        return OpenAI(api_key=self.API_KEY, base_url=self.BASE_URL)


# 从MySQL加载当天监控数据
# 从MySQL加载指定日期监控数据
def load_monitoring_data_from_mysql(date_str: str) -> pd.DataFrame:  # 修改：接收日期参数
    config = {
        'host': '1.92.135.70',
        'port': 3306,
        'user': 'root',
        'password': 'Aa123321',
        'database': 'AbnormalDetection',
        'charset': 'utf8mb4'
    }

    try:
        connection = pymysql.connect(**config)
        with connection.cursor() as cursor:
            # 修改：使用传入的日期参数而非当前日期
            sql = """
                SELECT id, cameraId, curTime, videoURL, info, type
                FROM warning
                WHERE DATE(curTime) = %s
            """
            cursor.execute(sql, (date_str,))  # 修改：传入日期参数
            rows = cursor.fetchall()
            if not rows:
                print(f"当天（{date_str}）无告警数据")  # 修复未定义变量today_str
                return pd.DataFrame(columns=['id', 'cameraId', 'curTime', 'videoURL', 'info', 'type'])
            df = pd.DataFrame(rows, columns=['id', 'cameraId', 'curTime', 'videoURL', 'info', 'type'])
            return df
    except Exception as e:
        print(f"MySQL 查询异常: {e}")
        return pd.DataFrame()
    finally:
        if 'connection' in locals() and connection:
            connection.close()


# 构建 Prompt
def build_cot_prompt(df: pd.DataFrame, date_str: str) -> str:  # 修改：接收日期参数
    # 修改：使用指定日期而非从数据中提取
    latest_date = datetime.strptime(date_str, '%Y-%m-%d').date()

    def generate_table(dataframe):
        columns = dataframe.columns.tolist()
        table = "| " + " | ".join(columns) + " |\n"
        table += "| " + " | ".join(["---"] * len(columns)) + " |\n"
        for _, row in dataframe.iterrows():
            table += "| " + " | ".join(str(x) for x in row) + " |\n"
        return table

    prompt_template = f"""你是一名安防监控分析专家，请基于下列原始事件数据，输出一份**结构化安全日报**。请围绕以下四类告警：

- **dangerous area**（危险区域侵入）
- **tumble**（人员摔倒）
- **stranger**（陌生人出现）
- **cheat**（非真人欺骗）

【原始数据】
{generate_table(df)}

【分析要求】
1. **事件分类**：分别统计四类事件数量，列出相关 cameraId 和时间。
2. **时间/空间分析**：识别事件高发时段与重点摄像头。
3. **风险评估**：
   - cheat > tumble > dangerous area > stranger（风险由高到低）
   - 同一摄像头频繁触发应提高风险等级
4. **建议措施**：针对每类事件提供可执行建议。

【输出格式】
# 安防监控日报（{latest_date}）

## 概览
| 类型 | 事件数 | 占比 |
| --- | --- | --- |
| dangerous area | {{dangerous_count}} | {{dangerous_pct}} |
| tumble | {{tumble_count}} | {{tumble_pct}} |
| stranger | {{stranger_count}} | {{stranger_pct}} |
| cheat | {{cheat_count}} | {{cheat_pct}} |

> 综合风险等级：**{{risk_level}}**

## 分类型事件详情
### dangerous area
- 共 {{dangerous_count}} 起
- 典型记录：cameraId 和时间列出几条代表性事件

### tumble
- 共 {{tumble_count}} 起
- ...

### stranger
- ...

### cheat
- ...

## 时间与空间分布
- 高发时段分析
- 高发摄像头分析

## 建议措施
- cheat：
- tumble：
- dangerous area：
- stranger：
"""
    return prompt_template


# 调用千帆模型生成日报
def call_qianfan(prompt: str, client: OpenAI, model: str = "ernie-4.0-turbo-8k") -> str:
    try:
        prompt = prompt.encode("utf-8").decode("utf-8")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用失败：{e}"


# 保存报告
def save_report(content: str, filepath: str):
    try:
        # 获取目录路径并创建（如果不存在）
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"保存报告失败: {str(e)}")
        return False


# 创建Flask应用
app = Flask(__name__)
# 只允许特定域名访问
CORS(app, resources={r"/*": {"origins": ["http://localhost:*", "http://127.0.0.1:*"]}})
# 添加doc参数指定Swagger文档路径
api = Api(app, version='1.0', title='安防监控日报API', description='生成和获取安防监控日报的API', doc='/swagger')
ns = api.namespace('reports', description='报告操作')

# 定义响应模型
success_model = api.model('SuccessResponse', {
    'status': fields.String(description='状态', example='success'),
    'message': fields.String(description='消息'),
    'date': fields.String(description='日期'),
    'content': fields.String(description='报告内容')
})

error_model = api.model('ErrorResponse', {
    'status': fields.String(description='状态', example='error'),
    'message': fields.String(description='错误消息')
})

# 报告生成接口
# 移除这行原生Flask路由装饰器
# @app.route('/generate_report', methods=['GET'])
@ns.route('/generate_report')
class GenerateReport(Resource):
    @api.doc(params={'date': '日期，格式YYYY-MM-DD'})
    @api.response(200, '成功', success_model)
    @api.response(400, '参数错误', error_model)
    @api.response(404, '数据不存在', error_model)
    @api.response(500, '服务器错误', error_model)
    def get(self):
        date_str = request.args.get('date')
        if not date_str:
            return jsonify({
                "status": "error", 
                "message": "缺少date参数，格式应为YYYY-MM-DD"
            }), 400

        try:
            # 验证日期格式
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({
                "status": "error", 
                "message": "日期格式错误，请使用YYYY-MM-DD"
            }), 400

        # 调用报告生成逻辑
        result, status_code = generate_report(date_str)
        return result, status_code

# 移除这行原生Flask路由装饰器
# @app.route('/get_report', methods=['GET'])
@ns.route('/get_report')
class GetReport(Resource):
    @api.doc(params={'date': '日期，格式YYYY-MM-DD'})
    @api.response(200, '成功', success_model)
    @api.response(400, '参数错误', error_model)
    @api.response(404, '文件不存在', error_model)
    @api.response(500, '服务器错误', error_model)
    def get(self):
        date_str = request.args.get('date')
        if not date_str:
            return {"status": "error", "message": "缺少date参数，格式应为YYYY-MM-DD"}, 400
        
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return {"status": "error", "message": "日期格式错误，请使用YYYY-MM-DD"}, 400
        
        report_path = f"./Daily/DailyReport/security_report_{date_str}.md"
        if not os.path.exists(report_path):
            return {"status": "error", "message": f"文件不存在: {report_path}"}, 404
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {
                "status": "success",
                "date": date_str,
                "content": content
            }, 200
        except Exception as e:
            return {"status": "error", "message": f"读取文件失败: {str(e)}"}, 500


# 报告生成逻辑封装
def generate_report(date_str):
    qianfan_config = QianfanConfig()
    client = qianfan_config.get_client()

    try:
        df = load_monitoring_data_from_mysql(date_str)
    except Exception as e:
        return {
            "status": "error", 
            "message": f"数据加载失败: {str(e)}"
        }, 500

    try:
        prompt = build_cot_prompt(df, date_str)
    except Exception as e:
        return {
            "status": "error", 
            "message": f"构建Prompt失败: {str(e)}"
        }, 500

    try:
        report_content = call_qianfan(prompt, client, model=qianfan_config.MODEL)
        if not report_content or report_content.startswith("调用失败："):
            return {
                "status": "error", 
                "message": f"模型调用失败: {report_content}"
            }, 500
    except Exception as e:
        return {
            "status": "error", 
            "message": f"调用失败: {str(e)}"
        }, 500

    report_filename = f"./Daily/DailyReport/security_report_{date_str}.md"
    if save_report(report_content, report_filename):
        return {
            "status": "success",
            "message": f"报告保存成功：{os.path.abspath(report_filename)}",
            "date": date_str,
            "content": report_content
        }, 200
    else:
        return {
            "status": "error", 
            "message": "报告保存失败"
        }, 500


# 服务启动入口
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9082, debug=True)
