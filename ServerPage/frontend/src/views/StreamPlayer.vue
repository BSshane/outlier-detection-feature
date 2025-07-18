<template>
  <div class="stream-container">
    <header class="header">
      <h1><i class="fas fa-video"></i> 智能视频推流</h1>
    </header>
    <main class="main-content">
      <div class="control-card">
        <div class="input-group">
          <div class="input-wrapper">
            <i class="fas fa-stream input-icon"></i>
            <input
              v-model="streamId"
              placeholder="输入视频流ID (例如: 1, camera1)"
              @keyup.enter="loadStream()"
              class="stream-input"
            >
          </div>
          <button @click="loadStream()" class="load-button">
            <i class="fas fa-play"></i> 加载
          </button>
        </div>
        <div class="recent-streams">
          <span v-for="stream in recentStreams" :key="stream" @click="loadStream(stream)" class="stream-tag">
            <i class="fas fa-history"></i> {{ stream }}
          </span>
        </div>
        <div class="status-message" v-if="statusMessage">
          <div :class="'alert alert-' + statusType">
            <i :class="statusIcon"></i> {{ statusMessage }}
          </div>
        </div>
      </div>
      <div class="video-card">
        <div class="video-wrapper">
          <div v-if="loading" class="loading-indicator">
            <div class="spinner"></div>
            <span>正在加载视频流...</span>
          </div>
          <img :src="videoUrl" alt="视频流" v-else class="video-frame" />
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue';
import axios from 'axios';

const streamId = ref('');
const videoUrl = ref('');
const loading = ref(false);
const statusMessage = ref('');
const statusType = ref('info');
const recentStreams = ref(['1', '2', 'camera1']);

const statusIcon = computed(() => ({
  'info': 'fas fa-info-circle',
  'success': 'fas fa-check-circle',
  'warning': 'fas fa-exclamation-circle',
  'error': 'fas fa-times-circle'
}[statusType.value]));

onMounted(() => {
  const urlParams = new URLSearchParams(window.location.search);
  const initialStream = urlParams.get('stream');
  if (initialStream) {
    streamId.value = initialStream;
    loadStream(initialStream);
  }
});

const loadStream = (id) => {
  const targetId = typeof id === 'string' ? id : streamId.value;
  if (!targetId) {
    statusMessage.value = '请输入视频流ID';
    statusType.value = 'warning';
    return;
  }
  loading.value = true;
  statusMessage.value = '';

  videoUrl.value = `http://1.92.135.70:9080/streams/video_feed/${targetId}`;

  const img = new Image();
  img.src = videoUrl.value;
  img.onload = () => {
    loading.value = false;
    statusMessage.value = '视频流加载成功';
    statusType.value = 'success';
    if (!recentStreams.value.includes(targetId)) {
      recentStreams.value.unshift(targetId);
      if (recentStreams.value.length > 5) recentStreams.value.pop();
    }
  };
  img.onerror = () => {
    loading.value = false;
    statusMessage.value = '视频流加载失败，请检查ID是否正确';
    statusType.value = 'error';
    videoUrl.value = '';
  };
};
</script>

<style scoped>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

.stream-container {
  max-width: 1200px;
  margin: 20px auto;
  padding: 20px;
  background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.header {
  text-align: center;
  margin-bottom: 30px;
}

.header h1 {
  font-size: 2.2rem;
  font-weight: 600;
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.main-content {
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: 20px;
}

.control-card {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.15);
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.input-group {
  display: flex;
  gap: 10px;
  align-items: center;
}

.input-wrapper {
  position: relative;
  flex: 1;
}

.input-icon {
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: #666;
}

.stream-input {
  width: 100%;
  padding: 10px 10px 10px 36px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  background: #fff;
  color: #000; /* 修改字体颜色为黑色 */
  transition: border-color 0.3s;
}

.stream-input:focus {
  outline: none;
  border-color: #2a5298;
  box-shadow: 0 0 0 3px rgba(42, 82, 152, 0.1);
}

.load-button {
  padding: 10px 20px;
  background: #2a5298;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: background 0.3s;
}

.load-button:hover {
  background: #1e3c72;
}

.recent-streams {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.stream-tag {
  padding: 6px 12px;
  background: #f0f2f5;
  border-radius: 16px;
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: background 0.3s;
  color: #333;
}

.stream-tag:hover {
  background: #e0e3e9;
}

.video-card {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.video-wrapper {
  position: relative;
  width: 100%;
  aspect-ratio: 9 / 16;
  max-height: 600px;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
}

.video-frame {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.status-message {
  margin-top: 15px;
}

.alert {
  padding: 12px;
  border-radius: 6px;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.alert-info { background: #e6f3ff; color: #2a5298; }
.alert-success { background: #e6ffed; color: #00a854; }
.alert-warning { background: #fff5e6; color: #ff7d00; }
.alert-error { background: #ffe6e6; color: #f53f3f; }

.loading-indicator {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background: rgba(0,0,0,0.7);
  color: white;
}

.spinner {
  width: 48px;
  height: 48px;
  border: 4px solid rgba(255,255,255,0.3);
  border-radius: 50%;
  border-top-color: #2a5298;
  animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@media (max-width: 768px) {
  .main-content {
    grid-template-columns: 1fr;
  }
  .video-wrapper {
    max-height: 400px;
  }
}
</style>