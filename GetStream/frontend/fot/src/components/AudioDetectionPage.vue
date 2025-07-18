<template>
  <div class="audio-detection-container">
    <div class="main-content">
      <h1 class="page-title">实时音频流检测</h1>
      <div class="content-wrapper-single-column">
        <div class="control-info-section full-width-section">
          <h2>检测控制与结果</h2>
          <div class="control-panel">
            <div class="input-group">
              <label for="camera-id">音频源ID:</label>
              <input type="text" id="camera-id" v-model="cameraId" placeholder="请输入数字ID" />
            </div>
            <button @click="startAnalysisFlow" class="action-button" :disabled="!cameraId || isAnalyzing">
              <span v-if="isAnalyzing">正在启动/检测...</span>
              <span v-else>播放并检测</span>
            </button>
            <button @click="stopAnalysis" class="action-button stop-button" :disabled="!isAnalyzing">
              停止获取
            </button>
          </div>

          <div class="results-panel info-panel">
            <h3>检测结果:</h3>
            <p v-if="detectionResult === null && !isAnalyzing" class="no-results">
              点击“播放并检测”按钮开始检测。
            </p>
            <div v-else-if="detectionResult?.status === 'detecting'" class="loading-state">
              <div class="spinner"></div>
              <p>正在检测中，请稍候...</p>
            </div>
            <div v-else-if="detectionResult?.status === 'success'" class="detection-success">
              <h4>检测完成！</h4>
              <!-- 检测结果以多行叙述的形式展示 -->
              <div class="result-text">
                <p>在 <span class="highlight">{{ detectionResult.data.timestamp }}</span> 的检测中</p>
                <p>音频被识别为 "<span class="highlight">{{ detectionResult.data.label }}</span>"</p>
                <p>置信度为 <span class="highlight">{{ (detectionResult.data.score * 100).toFixed(2) }}%</span> </p>
                <p>当前状态：<span
                    :class="{ 'danger-status': detectionResult.data.is_danger, 'safe-status': !detectionResult.data.is_danger }">
                    {{ detectionResult.data.is_danger ? '存在危险' : '安全' }}
                  </span> </p>
              </div>
            </div>
            <div v-else-if="detectionResult?.status === 'error'" class="detection-error">
              <h4>检测失败！</h4>
              <p>错误信息: {{ detectionResult.message }}</p>
            </div>
            <p v-else-if="!isAnalyzing && detectionResult !== null" class="no-results">
              分析已停止。
            </p>
          </div>

          <!-- 危险记录表格 -->
          <div class="danger-log-section">
            <h2>危险记录
              <button @click="clearDangerLogs" class="clear-button" :disabled="dangerLogs.length === 0">清除</button>
            </h2>
            <table class="danger-table">
              <thead>
                <tr>
                  <th>发生时间</th>
                  <th>类型</th>
                </tr>
              </thead>
              <tbody>
                <tr v-if="dangerLogs.length === 0">
                  <td colspan="2" class="no-logs">暂无危险记录。</td>
                </tr>
                <tr v-for="log in dangerLogs" :key="log.id">
                  <td>{{ log.startTime }}</td>
                  <td>{{ log.type }}</td>
                </tr>
              </tbody>
            </table>
          </div>

        </div>
      </div>
    </div>

    <CustomMessage v-if="showMessage" :message="messageContent" :type="messageType" :duration="3000"
      @close="showMessage = false" />
  </div>
</template>

<script>
import CustomMessage from '@/components/CustomMessage.vue';
import { audioMonitorApi } from '@/api/axios.js';

export default {
  name: 'AudioDetectionPage',
  components: {
    CustomMessage
  },
  data() {
    return {
      cameraId: '', // 用于绑定输入框的音频源ID
      detectionResult: null,
      streamError: false, // 标识后端流连接是否出错
      isAnalyzing: false,
      analysisIntervalId: null,
      fetchInterval: 1000, // 数据获取间隔保持1秒

      dangerLogs: [], // 存储危险记录的数组
      currentDangerSession: null, // 当前活跃的危险会话
      dangerThresholdSeconds: 5, // 5秒内记录为一次报警信息

      showMessage: false,
      messageContent: '',
      messageType: 'info',
    };
  },
  mounted() {
    console.log(`音频检测系统连接到: ${audioMonitorApi.defaults.baseURL}`);
  },
  beforeUnmount() {
    this.stopAnalysis();
  },
  methods: {
    displayMessage(content, type = 'info') {
      this.messageContent = content;
      this.messageType = type;
      this.showMessage = true;
    },

    async startAnalysisFlow() {
      if (!this.cameraId) {
        this.displayMessage('请输入音频源ID！', 'warning');
        return;
      }

      this.isAnalyzing = true;
      this.detectionResult = { status: 'detecting' };
      this.streamError = false; // 重置流错误状态

      // 1. 每次开始前，先尝试停止当前的后端分析
      await this.stopAnalysisBackend();

      // 构建要发送给后端进行分析的流地址
      // 假设后端现在直接接收摄像头ID作为 rtmp_url 的一部分
      // IMPORTANT: 请根据您的后端实际接收参数的方式调整此处的 rtmp_url 构造
      const rtmpUrlForBackend = `${this.cameraId}`;
      console.log(`开始向后端请求分析音频流: ${rtmpUrlForBackend}`);

      // 2. 启动后端分析
      try {
        const response = await audioMonitorApi.post('/monitor/start', {
          rtmp_url: rtmpUrlForBackend, // 将构建好的 URL 发送给后端
        });
        if (response.status === 200) {
          this.displayMessage('后端分析已成功启动！', 'success');
          this.startFetchingResults();
        } else {
          this.displayMessage('启动后端分析失败: ' + (response.data.message || '未知错误'), 'error');
          this.isAnalyzing = false;
        }
      } catch (error) {
        console.error('调用 /monitor/start 失败:', error);
        this.displayMessage('启动后端分析请求失败: ' + (error.response?.data?.message || error.message || '请检查后端服务。'), 'error');
        this.isAnalyzing = false;
        this.streamError = true; // 标记流连接失败
        this.detectionResult = { status: 'error', message: '后端流连接失败或无法启动分析。' };
      }
    },

    async stopAnalysisBackend() {
      try {
        const response = await audioMonitorApi.post('/monitor/stop');
        if (response.status === 200) {
          this.displayMessage('后端分析已成功停止。', 'info');
        } else if (response.status === 400 && response.data.message === "没有活动的分析") {
            console.log('没有活动的后端分析可停止。');
        } else {
          this.displayMessage('停止后端分析失败: ' + (response.data.message || '未知错误'), 'warning');
        }
      } catch (error) {
        if (error.response && error.response.status === 400 && error.response.data.message === "没有活动的分析") {
            console.log('没有活动的后端分析可停止。');
        } else {
            console.error('调用 /monitor/stop 失败:', error);
            this.displayMessage('停止后端分析请求失败: ' + (error.response?.data?.message || error.message || '请检查后端服务。'), 'error');
        }
      }
    },

    startFetchingResults() {
      this.stopFetchingResults();
      this.analysisIntervalId = setInterval(async () => {
        try {
          const response = await audioMonitorApi.get('/monitor/analyze');
          if (response.status === 200) {
            this.detectionResult = { status: 'success', data: response.data };
            const { is_danger, timestamp, label } = response.data;
            const currentTime = new Date(timestamp).getTime(); // 将时间戳转换为毫秒

            if (is_danger) {
              if (!this.currentDangerSession) {
                // 开始一个新的危险会话
                this.currentDangerSession = {
                  id: Date.now(), // 唯一ID
                  startTime: timestamp,
                  endTime: timestamp, // 初始结束时间与开始时间相同
                  type: label,
                };
                this.dangerLogs.push(this.currentDangerSession);
              } else {
                // 继续现有的危险会话
                const lastLogEntry = this.dangerLogs[this.dangerLogs.length - 1];
                const lastLogEndTime = new Date(lastLogEntry.endTime).getTime();

                // 如果当前危险在上次记录的危险事件结束时间的5秒内
                if (currentTime - lastLogEndTime <= this.dangerThresholdSeconds * 1000) {
                  lastLogEntry.endTime = timestamp; // 延长上次记录的结束时间
                } else {
                  // 超出5秒窗口，开始新的危险会话
                  this.currentDangerSession = {
                    id: Date.now(),
                    startTime: timestamp,
                    endTime: timestamp,
                    type: label,
                  };
                  this.dangerLogs.push(this.currentDangerSession);
                }
              }
            } else {
              // 不危险，检查是否有活跃的危险会话刚刚结束
              if (this.currentDangerSession) {
                // 结束最后一个危险会话 (endTime 已经更新到最后一次危险发生的时间)
                this.currentDangerSession = null; // 重置会话
              }
            }

          } else if (response.status === 400 && response.data.message === "没有活动的分析流") {
            this.displayMessage('后端分析流已终止。', 'info');
            this.stopAnalysis();
          } else {
            this.displayMessage('获取分析结果失败: ' + (response.data.message || '未知错误'), 'warning');
            this.detectionResult = { status: 'error', message: '获取结果失败。' };
          }
        } catch (error) {
          console.error('获取分析结果失败:', error);
          this.detectionResult = { status: 'error', message: '获取结果请求失败: ' + (error.response?.data?.message || error.message || '请检查后端服务或网络。') };
          this.stopAnalysis();
        }
      }, this.fetchInterval);
    },

    stopFetchingResults() {
      if (this.analysisIntervalId) {
        clearInterval(this.analysisIntervalId);
        this.analysisIntervalId = null;
      }
    },

    async stopAnalysis() {
      this.stopFetchingResults();
      await this.stopAnalysisBackend();
      this.isAnalyzing = false;
      this.detectionResult = null;
      this.streamError = false; // 停止时重置错误状态

      // 当分析停止时，重置任何正在进行的危险会话
      this.currentDangerSession = null;
      this.displayMessage('音频分析已完全停止。', 'info');
    },

    // 新增清除危险记录的方法
    clearDangerLogs() {
      this.dangerLogs = [];
      this.currentDangerSession = null;
      this.displayMessage('危险记录已清除。', 'info');
    }
  },
  watch: {
    // 监听 cameraId 变化；如果被清空，则重置相关状态并停止分析
    cameraId(newVal) {
      if (!newVal) {
        this.detectionResult = null;
        this.streamError = false;
        this.stopAnalysis();
        this.dangerLogs = []; // 清空危险记录
        this.currentDangerSession = null; // 重置危险会话
      }
    }
  }
};
</script>

<style scoped>
/* 按钮样式 */
.action-button.stop-button {
  background-color: #dc3545;
  margin-top: 15px;
}

.action-button.stop-button:hover:not(:disabled) {
  background-color: #c82333;
}

/* 整体容器布局 */
.audio-detection-container {
  min-height: 100vh;
  background-color: #f0f2f5;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 20px; /* 增加整体内边距 */
}

/* 主内容区域 */
.main-content {
  flex-grow: 1;
  padding: 20px 30px;
  max-width: 800px; /* 限制最大宽度 */
  width: 100%; /* 确保在最大宽度内占满 */
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

/* 页面标题 */
.page-title {
  color: #2c3e50;
  margin-bottom: 25px;
  text-align: center;
  font-size: 2.2em;
  font-weight: 700;
}

/* 单列内容包装器 */
.content-wrapper-single-column {
  display: flex;
  flex-direction: column;
  gap: 25px; /* 各部分之间的间距 */
  width: 100%;
}

/* 控制和信息部分 */
.control-info-section.full-width-section {
  flex: none;
  min-width: unset;
  width: 100%;
  padding: 0;
  box-shadow: none;
  background-color: transparent;
}

.control-info-section h2 {
  color: #34495e;
  margin-top: 0;
  margin-bottom: 15px;
  font-size: 1.5em;
}

/* 控制面板 */
.control-panel {
  padding: 20px;
  background-color: #f9f9f9;
  border-radius: 6px;
  margin-bottom: 20px;
  flex-shrink: 0;
}

.input-group {
  margin-bottom: 15px;
}

.input-group label {
  display: block;
  margin-bottom: 5px;
  color: #555;
  font-weight: bold;
}

.input-group input {
  width: 100%;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-sizing: border-box;
}

.action-button {
  width: 100%;
  padding: 12px 20px;
  background-color: #3498db;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1.1em;
  transition: background-color 0.3s ease;
}

.action-button:hover:not(:disabled) {
  background-color: #2980b9;
}

.action-button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

/* 结果面板 */
.results-panel {
  flex-grow: 1;
  background-color: #f0f8ff;
  border: 1px solid #cce5ff;
  border-radius: 8px;
  padding: 25px;
  min-height: 150px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  margin-top: 0;
}

.results-panel h3 {
  margin-top: 0;
}

.no-results {
  color: #666;
  font-style: italic;
  font-size: 1.1em;
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 15px;
}

.spinner {
  border: 5px solid #f3f3f3;
  border-top: 5px solid #3498db;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-state p {
  font-size: 1.2em;
  color: #3498db;
  font-weight: bold;
}

.detection-success {
  color: #28a745;
  font-weight: bold;
  text-align: left;
  width: 100%;
}

/* 检测结果文本样式 */
.detection-success .result-text {
  background-color: #e6ffe6;
  border: 1px solid #4CAF50;
  padding: 15px;
  border-radius: 5px;
  white-space: pre-wrap; /* 保持换行 */
  word-break: break-all;
  font-size: 0.95em;
  line-height: 1.6;
  max-height: 300px;
  overflow-y: auto;
  color: #333;
}

.detection-success .result-text p {
  margin: 0; /* 移除段落默认外边距 */
  padding-bottom: 5px; /* 增加行间距 */
}

/* 关键信息高亮 */
.highlight {
  color: #007bff; /* 蓝色 */
  font-weight: bold;
}

/* 安全/危险状态颜色 */
.danger-status {
  color: #dc3545; /* 红色 */
  font-weight: bold;
}

.safe-status {
  color: #28a745; /* 绿色 */
  font-weight: bold;
}

/* 错误信息显示 */
.detection-error {
  color: #dc3545;
  font-weight: bold;
  text-align: center;
}

.detection-error p {
  background-color: #ffe6e6;
  border: 1px solid #dc3545;
  padding: 15px;
  border-radius: 5px;
  font-size: 1.05em;
}

/* 危险记录表格样式 */
.danger-log-section {
  margin-top: 30px;
  padding: 20px;
  background-color: #fff3e0; /* 浅橙色背景，表示警告 */
  border: 1px solid #ffcc80;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  position: relative; /* 用于定位清除按钮 */
}

.danger-log-section h2 {
  color: #e65100; /* 深橙色标题 */
  margin-top: 0;
  margin-bottom: 20px;
  text-align: center;
  font-size: 1.6em;
}

/* 清除按钮样式 */
.danger-log-section .clear-button {
  position: absolute;
  top: 15px;
  right: 15px;
  background-color: #f44336; /* 红色 */
  color: white;
  border: none;
  border-radius: 5px;
  padding: 8px 12px;
  cursor: pointer;
  font-size: 0.9em;
  transition: background-color 0.3s ease;
}

.danger-log-section .clear-button:hover:not(:disabled) {
  background-color: #d32f2f;
}

.danger-log-section .clear-button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}


.danger-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 15px;
}

.danger-table th,
.danger-table td {
  border: 1px solid #ffcc80;
  padding: 10px;
  text-align: left;
}

.danger-table th {
  background-color: #ffb74d; /* 橙色表头 */
  color: white;
  font-weight: bold;
}

.danger-table tr:nth-child(even) {
  background-color: #fff8e1; /* 隔行背景色 */
}

.danger-table .no-logs {
  text-align: center;
  font-style: italic;
  color: #888;
  padding: 20px;
}

/* 媒体查询，适配小屏幕 */
@media (max-width: 1024px) {
  .main-content {
    padding: 15px;
  }
  .control-info-section.full-width-section {
    padding: 0;
  }
}
</style>
