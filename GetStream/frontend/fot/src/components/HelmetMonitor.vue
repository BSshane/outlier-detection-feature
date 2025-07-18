<template>
  <div class="risk-monitoring-container">
    <div class="main-content">
      <h1 class="page-title">实时异常行为监控系统</h1>

      <div class="content-wrapper">
        <!-- 左侧：视频流与控制区 -->
        <div class="video-section">
          <h2>实时监控画面</h2>
          <div class="video-player">
            <!-- 视频流通过 img 标签显示 -->
            <img :src="videoFeedUrl" alt="实时检测画面" class="responsive-video" @error="handleVideoError">
            <!-- 视频加载失败时的提示 -->
            <div v-if="videoError" class="video-placeholder error-text">
              <p>视频流加载失败。<br>请检查后端服务是否已启动。</p>
            </div>
          </div>
          <!-- 危险区域控制面板 -->
          <div class="control-panel">
            <h3>设置危险区域</h3>
            <div class="input-grid">
              <div class="input-group">
                <label for="x1">点1 (x, y):</label>
                <input type="number" id="x1" v-model.number="dangerZone.x1">
                <input type="number" id="y1" v-model.number="dangerZone.y1">
              </div>
              <div class="input-group">
                <label for="x2">点2 (x, y):</label>
                <input type="number" id="x2" v-model.number="dangerZone.x2">
                <input type="number" id="y2" v-model.number="dangerZone.y2">
              </div>
            </div>
            <button @click="setDangerZone" class="action-button">应用区域设置</button>
            <p v-if="dangerZoneStatus" class="status-message">{{ dangerZoneStatus }}</p>
          </div>
        </div>

        <!-- 右侧：信息展示区 -->
      </div>
    </div>
  </div>
</template>

<script>
// 建议使用一个统一的API客户端，例如axios
import axios from 'axios';

export default {
  name: 'RiskMonitoring',
  data() {
    return {
      // 视频流地址，指向您的后端Flask路由
      videoFeedUrl: 'http://127.0.0.1:9086/video_feed', // 请确保端口正确
      videoError: false,

      // 危险区域坐标，使用响应式对象管理
      dangerZone: {
        x1: 100,
        y1: 100,
        x2: 700,
        y2: 700,
      },
      dangerZoneStatus: '', // 设置危险区域后的状态消息
    };
  },
  methods: {
    // 设置危险区域
    async setDangerZone() {
      this.dangerZoneStatus = '正在设置...';
      try {
        const response = await axios.post('http://127.0.0.1:9086/set_danger_zone', this.dangerZone);
        this.dangerZoneStatus = response.data.message;

        // 刷新视频流以应用新设置（通过添加时间戳来强制浏览器重新加载）
        this.videoFeedUrl = `http://127.0.0.1:9086/video_feed?timestamp=${new Date().getTime()}`;
      } catch (error) {
        console.error('设置危险区域失败:', error);
        this.dangerZoneStatus = '设置失败，请检查后端服务。';
      } finally {
        // 3秒后清除状态消息
        setTimeout(() => {
          this.dangerZoneStatus = '';
        }, 3000);
      }
    },


    // 处理视频加载错误
    handleVideoError() {
      this.videoError = true;
    }
  }
};
</script>

<style scoped>
/* 整体布局和样式，借鉴了您提供的参考风格 */
.risk-monitoring-container {
  min-height: 100vh;
  background-color: #f0f2f5;
  padding: 20px 30px;
}
.page-title {
  color: #2c3e50;
  margin-bottom: 25px;
  text-align: center;
  font-size: 2.2em;
  font-weight: 700;
}
.content-wrapper {
  display: flex;
  gap: 25px;
  justify-content: center; /* 让视频区居中 */
}

/* 左侧区域 */
.video-section {
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  padding: 20px;
  width: 80%;
}
.video-section h2, .info-section h3 {
  color: #34495e;
  margin-top: 0;
  margin-bottom: 15px;
  font-size: 1.5em;
  border-bottom: 2px solid #e0e0e0;
  padding-bottom: 10px;
}
.video-player {
  width: 100%;
  background-color: #000;
  border-radius: 6px;
  overflow: hidden;
  position: relative;
  /* 保持16:9的宽高比 */
  padding-top: 56.25%;
}
.responsive-video {
  position: absolute;
  top: 0; left: 0; width: 100%; height: 100%;
  object-fit: contain; /* 使用 contain 保证视频完整显示 */
}
.video-placeholder {
  position: absolute;
  top: 0; left: 0; width: 100%; height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #bbb;
}
.error-text {
  color: #e74c3c;
}

/* 控制面板 */
.control-panel {
  margin-top: 20px;
  padding: 15px;
  background-color: #f9fafb;
  border-radius: 6px;
  border: 1px solid #e5e7eb;
}
.control-panel h3 {
  font-size: 1.2em;
  border: none;
  padding-bottom: 0;
}
.input-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 15px;
  margin: 15px 0;
}
.input-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
.input-group label {
  font-weight: 500;
  color: #333;
}
.input-group input {
  width: 80px;
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
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
.action-button:hover {
  background-color: #2980b9;
}
.status-message {
  margin-top: 10px;
  text-align: center;
  color: #27ae60;
  font-weight: bold;
}

/* 右侧信息区 */
.info-section {
  flex: 2;
  min-width: 350px;
  display: flex;
  flex-direction: column;
  gap: 25px;
}
.info-panel {
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  padding: 20px;
  display: flex;
  flex-direction: column;
}
.risk-pool {
  flex-grow: 1; /* 让风险池面板填充剩余空间 */
}
.count-badge {
  color: #3498db;
  font-weight: bold;
}
.list-container {
  overflow-y: auto;
  flex-grow: 1;
}
.empty-text {
  text-align: center;
  color: #999;
  margin-top: 20px;
}
.list-item {
  padding: 12px;
  border-bottom: 1px solid #f0f0f0;
  animation: fadeIn 0.5s ease-in-out;
}
.list-item:last-child {
  border-bottom: none;
}
.risk-item.new {
  background-color: #fffbe6; /* 淡黄色高亮新风险 */
}
.item-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 5px;
  font-size: 0.9em;
  color: #555;
}
.item-body {
  font-size: 1em;
  color: #333;
}

/* 状态颜色 */
.status-new { color: #f39c12; font-weight: bold; }
.status-processed { color: #27ae60; }
.status-running { color: #27ae60; font-weight: bold; }
.status-stopped { color: #e74c3c; font-weight: bold; }

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
