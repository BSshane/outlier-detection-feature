<template>
  <div class="fall-detection-container">
    <div class="main-content">
      <h1 class="page-title">实时跌倒检测监控</h1>

      <div class="content-wrapper">
        <div class="video-section">
          <h2>实时检测画面</h2>
          <div class="video-player">
            <img
              v-if="videoFeedUrl"
              :src="videoFeedUrl"
              :style="{ maxWidth: `${config.max_width}px` }"
              alt="Real-time Detection Feed"
              class="responsive-video"
              @error="handleVideoError"
              :key="imageKey" />
            <div v-else class="video-placeholder">
              <p>{{ videoLoadingText }}</p>
            </div>
            <div v-if="videoError" class="video-placeholder error-text">
              <p>{{ videoErrorText }}</p>
            </div>
          </div>
        </div>

        <div class="control-info-section">
          <h2>检测状态</h2>
          <div class="control-panel">
            <p>{{ statusText }}</p>
            <div v-if="fallDetected" class="alert-box">
              <p class="alert-text">⚠️ 检测到跌倒事件 ⚠️</p>
              <p class="alert-time">{{ lastFallTime || '刚刚' }}</p>
            </div>
          </div>
          <div class="info-panel">
            <h3>系统信息</h3>
            <p>当前状态: {{ systemStatus }}</p>
            <p>检测模式: 人体姿态分析</p>
            <p>摄像头: 1</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';
import { ElMessage } from 'element-plus';

export default {
  name: 'FallDetectionPage',
  data() {
    return {
      backendUrl: 'http://localhost:9089', // Flask 后端地址
      config: {
        title: '跌倒检测系统',
        max_width: 800,
        status_text: '系统运行中...'
      },
      videoFeedUrl: null,
      videoError: false,
      videoErrorText: '',
      videoLoadingText: '正在连接视频流...',
      statusText: '系统初始化中...',
      fallDetected: false,
      lastFallTime: null,
      systemStatus: '正在连接',
      imageKey: 0,
      retryCount: 0,
      maxRetries: 5
    };
  },
  mounted() {
    console.log(`跌倒检测系统连接到: ${this.backendUrl}`);
    this.initializeVideoFeed();
  },
  beforeUnmount() {
    // 清理资源
    this.videoFeedUrl = null;
  },
  methods: {
    initializeVideoFeed() {
      // 直接使用后端提供的视频流接口
      this.videoFeedUrl = `${this.backendUrl}/video_feed`;
      this.statusText = '正在连接视频流...';
      this.systemStatus = '正在连接';
      this.retryCount = 0;
    },
    handleVideoError() {
      this.videoError = true;
      this.videoFeedUrl = null;

      if (this.retryCount < this.maxRetries) {
        this.retryCount++;
        this.videoErrorText = `视频连接失败，尝试重新连接 (${this.retryCount}/${this.maxRetries})...`;
        this.statusText = '视频连接中断，正在重试...';

        setTimeout(() => {
          this.initializeVideoFeed();
        }, 2000);
      } else {
        this.videoErrorText = '视频连接失败，请检查后端服务是否正常运行。';
        this.statusText = '视频连接失败，请刷新页面或联系管理员。';
        this.systemStatus = '连接失败';
      }
    },
    // 模拟检测状态更新（根据实际需要可以通过WebSocket等方式实现）
    updateDetectionStatus(detected, time = null) {
      this.fallDetected = detected;
      this.lastFallTime = time;

      if (detected) {
        this.statusText = '⚠️ 检测到跌倒事件 ⚠️';
        this.systemStatus = '告警中';
        ElMessage.error('检测到跌倒事件！');
      } else {
        this.statusText = '系统运行正常，未检测到异常';
        this.systemStatus = '正常运行';
      }
    }
  }
};
</script>

<style scoped>
/* 保持原有样式不变 */
.fall-detection-container {
  min-height: 100vh;
  background-color: #f0f2f5;
}

.main-content {
  flex-grow: 1;
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
  flex-wrap: wrap;
}

.video-section {
  flex: 3;
  min-width: 400px;
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  padding: 20px;
}

.video-section h2,
.control-info-section h2 {
  color: #34495e;
  margin-top: 0;
  margin-bottom: 15px;
  font-size: 1.5em;
}

.video-player {
  width: 100%;
  background-color: #000;
  border-radius: 6px;
  overflow: hidden;
  position: relative;
  padding-top: 75%; /* 4:3 宽高比 */
}

.responsive-video {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain; /* 改为 contain 以显示完整图片 */
}

.video-placeholder {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #333;
  color: #bbb;
  text-align: center;
  font-size: 1.2em;
}

.error-text {
  color: #e74c3c;
}

.control-info-section {
  flex: 2;
  min-width: 300px;
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  padding: 20px;
}

.control-panel {
  padding: 20px;
  background-color: #f9f9f9;
  border-radius: 6px;
  margin-bottom: 20px;
}

.info-panel h3 {
  margin-top: 0;
  color: #34495e;
}

.info-panel p {
  color: #555;
}

.alert-box {
  background-color: #ffebee;
  border-left: 4px solid #e53935;
  padding: 10px;
  margin-top: 15px;
  border-radius: 4px;
}

.alert-text {
  color: #e53935;
  font-weight: bold;
  margin-bottom: 5px;
}

.alert-time {
  color: #666;
  font-size: 0.9em;
}
</style>