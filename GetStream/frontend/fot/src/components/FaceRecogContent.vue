<template>
  <div class="face-recognition-container">
    <div class="main-content">
      <h1 class="page-title">实时人脸识别监控</h1>

      <div class="content-wrapper">
        <!-- 视频流显示区 -->
        <div class="video-section">
          <h2>实时监控画面</h2>
          <div class="video-player">
            <!-- 只有当 videoFeedUrl 有效时才显示图像 -->
            <img v-if="videoFeedUrl" :src="videoFeedUrl" alt="Live Video Feed" class="responsive-video" @error="handleVideoError">
            <!-- 占位符，提示用户操作 -->
            <div v-else class="video-placeholder">
              <p>请输入摄像头ID并点击“加载视频”</p>
            </div>
            <div v-if="videoError" class="video-placeholder error-text">
              <p>视频流加载失败。<br>请检查摄像头ID是否正确以及后端服务是否启动。</p>
            </div>
          </div>
        </div>

        <!-- 控制和信息区 -->
        <div class="control-info-section">
          <h2>监控控制</h2>
          <div class="control-panel">
            <div class="input-group">
              <label for="stream-id">摄像头ID:</label>
              <!-- 输入框，用于输入 stream_id -->
              <input type="text" id="stream-id" v-model="streamId" placeholder="例如: 1">
            </div>
            <!-- 加载视频的按钮 -->
            <button @click="loadVideoStream" class="action-button" :disabled="!streamId">
              加载视频
            </button>
          </div>
          <div class="info-panel">
            <h3>说明</h3>
            <p>本系统通过后端算法实时分析视频流：</p>
            <ul>
              <li><span class="dot known"></span><span class="label-text">绿色框:</span> 识别为真人</li>
              <li><span class="dot unknown"></span><span class="label-text">红色框:</span> 检测为欺骗攻击 (非真人)</li>
              <li>识别出的人名或"unknown"会直接显示在视频框内。</li>
              <li>"陌生人"或"欺骗"告警会自动写入后端数据库。</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { faceRecognitionApi } from '@/api/axios.js';
import { ElMessage } from 'element-plus';

export default {
  name: 'FaceRecognitionPage',
  data() {
    return {
      // 从我们配置的 API 实例中获取后端地址
      backendUrl: faceRecognitionApi.defaults.baseURL,
      streamId: '1', // 默认的摄像头ID
      videoFeedUrl: '', // 动态构建的完整视频流URL，初始为空
      videoError: false,
    };
  },
  mounted() {
    // 组件挂载后，显示它将使用的后端 URL
    const backendUrl = faceRecognitionApi.defaults.baseURL;
    console.log(`人脸识别系统连接到: ${backendUrl}`);

  },
  methods: {
    loadVideoStream() {
      if (!this.streamId) {
        ElMessage.warning('请输入摄像头ID！');
        return;
      }
      // 重置错误状态
      this.videoError = false;
      // 动态构建视频流的完整 URL
      this.videoFeedUrl = `${this.backendUrl}/video_feed/${this.streamId}`;
      console.log(`正在尝试加载视频流: ${this.videoFeedUrl}`);
      ElMessage.info(`正在加载摄像头 ${this.streamId} 的视频流...`);
    },
    handleVideoError() {
      console.error('视频流加载失败');
      this.videoError = true;
      // 加载失败时，清空URL，这样可以再次显示提示信息
      this.videoFeedUrl = '';
    }
  }
};
</script>

<style scoped>
/* 样式与之前类似，但针对新布局进行微调 */
.face-recognition-container {
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

/* 视频区 */
.video-section {
  flex: 3;
  min-width: 400px;
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  padding: 20px;
}

.video-section h2, .control-info-section h2 {
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
  padding-top: 75%; /* 4:3 apect ratio */
}

.responsive-video {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.video-placeholder {
  position: absolute;
  top: 0; left: 0; width: 100%; height: 100%;
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

/* 控制和信息区 */
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
  box-sizing: border-box; /* 保证 padding 不会撑大宽度 */
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

.info-panel h3 {
  margin-top: 0;
  color: #34495e;
}
.info-panel ul {
  list-style: none;
  padding-left: 0;
}
.info-panel li {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
  color: #555;
}
.dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-right: 10px;
  flex-shrink: 0;
}
.dot.known { background-color: #27ae60; }
.dot.unknown { background-color: #e74c3c; }
.label-text {
  font-weight: bold;
  margin-right: 5px;
}

</style>
