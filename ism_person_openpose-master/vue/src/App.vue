<script setup>
import { ref, onMounted, onUnmounted } from 'vue';

// 响应式变量
const latestImage = ref(''); // 用于存储最新图片路径
const statusMessage = ref('系统正在初始化...'); // 用于显示状态信息
const pageTitle = ref('跌倒检测系统'); // 页面标题
const maxWidth = ref(800); // 图片最大宽度
const refreshRate = ref(3000); // 刷新频率 (毫秒)

let intervalId = null; // 用于存储定时器的ID

// 从 Flask 后端获取配置信息
const fetchConfig = async () => {
  try {
    const response = await fetch('http://localhost:5000/api/config'); // 访问 Flask 的配置 API
    const data = await response.json();
    pageTitle.value = data.title;
    refreshRate.value = data.refresh_rate;
    maxWidth.value = data.max_width;
    statusMessage.value = data.status_text; // 设置初始状态文本
  } catch (error) {
    console.error('获取配置时出错:', error);
    statusMessage.value = '无法获取系统配置。';
  }
};

// 从 Flask 后端获取最新图片
const fetchLatestImage = async () => {
  try {
    const response = await fetch('http://localhost:5000/get_latest_image'); // 访问 Flask 的图片 API
    const data = await response.json();

    if (data.image_path) {
      latestImage.value = `http://localhost:5000${data.image_path}`; // 拼接完整的图片 URL
      statusMessage.value = data.status;
    } else {
      statusMessage.value = data.status;
      latestImage.value = ''; // 没有图片时清除
    }
  } catch (error) {
    console.error('获取图片时出错:', error);
    statusMessage.value = '无法连接到后端或获取图像。';
    latestImage.value = ''; // 清除图片以防显示旧图
  }
};

// 组件挂载时启动定时器
onMounted(async () => {
  // 首先获取配置
  await fetchConfig();

  // 首次加载立即获取一次图片
  fetchLatestImage();

  // 设置定时器，按 refreshRate 刷新
  // 确保 refreshRate 已经被 fetchConfig 更新
  intervalId = setInterval(fetchLatestImage, refreshRate.value);
});

// 组件卸载时清除定时器，避免内存泄漏
onUnmounted(() => {
  if (intervalId) {
    clearInterval(intervalId);
  }
});

// 从 detect.py 中获取的 RTMP 帧间隔信息
// 这是一个固定值，可以直接在前端显示
const rtmpFrameInterval = 30; // 从 detect.py 中得知
</script>

<template>
  <div class="app-container">
    <header>
      <h1>{{ pageTitle }}</h1>
      <p class="status-message">{{ statusMessage }}</p>
      <!-- 根据你的 detect.py，RTMP流的帧间隔是固定的 30 帧 -->
      <p class="rtmp-info">RTMP流：每 {{ rtmpFrameInterval }} 帧保存一张图片进行显示。</p>
    </header>

    <main>
      <div v-if="latestImage" class="image-display">
        <!-- 动态绑定图片路径和最大宽度 -->
        <img :src="latestImage" :style="{ maxWidth: maxWidth + 'px' }" alt="检测图像" />
      </div>
      <div v-else class="no-image-placeholder">
        <p>等待检测图像...</p>
      </div>
    </main>
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem;
  font-family: 'Inter', sans-serif; /* 使用 Inter 字体 */
  background-color: #f0f2f5;
  min-height: 100vh;
  box-sizing: border-box; /* 确保 padding 不会增加总宽度 */
}

header {
  text-align: center;
  margin-bottom: 2rem;
  background-color: #ffffff;
  padding: 1.5rem;
  border-radius: 12px; /* 增加圆角 */
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); /* 增加阴影 */
  width: 100%;
  max-width: 900px;
}

h1 {
  color: #333;
  margin-bottom: 0.5rem;
  font-size: 2.2rem; /* 调整标题大小 */
}

.status-message {
  color: #555;
  font-size: 1.1rem;
  margin-top: 0.5rem;
}

.rtmp-info {
  color: #777;
  font-size: 0.9rem;
  margin-top: 0.5rem;
}

main {
  width: 100%;
  max-width: 900px;
  background-color: #ffffff;
  padding: 1.5rem;
  border-radius: 12px; /* 增加圆角 */
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); /* 增加阴影 */
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
  overflow: hidden; /* 防止图片溢出 */
}

.image-display {
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.image-display img {
  width: 100%; /* 确保图片在其容器内缩放 */
  height: auto;
  display: block;
  border-radius: 8px; /* 图片也有圆角 */
  object-fit: contain; /* 确保图片完整显示，不裁剪 */
}

.no-image-placeholder {
  text-align: center;
  color: #888;
  font-style: italic;
  font-size: 1.2rem;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .app-container {
    padding: 1rem;
  }

  header, main {
    padding: 1rem;
    border-radius: 8px;
  }

  h1 {
    font-size: 1.8rem;
  }

  .status-message {
    font-size: 1rem;
  }

  .rtmp-info {
    font-size: 0.8rem;
  }
}
</style>
