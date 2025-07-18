<template>
  <div style="padding: 20px; max-width: 800px; margin: 0 auto;">
    <h1>人脸识别展示</h1>
    <div style="display: flex; align-items: center; gap: 10px;">
      <input
        v-model="videoInputUrl"
        placeholder="请输入 RTMP URL（如 rtmp://1.92.135.70:9090/live/1）"
        style="flex: 1; padding: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 4px;"
      />
      <button
        @click="loadVideo"
        style="padding: 10px 20px; font-size: 16px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;"
        :disabled="loading"
      >
        {{ loading ? '加载中...' : '加载视频' }}
      </button>
    </div>
    <div v-if="videoUrl" style="margin-top: 20px;">
      <video controls :src="videoUrl" style="max-width: 100%; border: 1px solid #ccc;" autoplay>
        您的浏览器不支持视频播放。
      </video>
    </div>
    <div v-if="error" style="color: red; margin-top: 10px; font-size: 14px;">
      {{ error }}
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'App',
  data() {
    return {
      videoInputUrl: '', // 输入的 RTMP URL
      videoUrl: '',      // 实际播放的 HLS URL
      error: '',         // 错误信息
      loading: false     // 加载状态
    };
  },
  methods: {
    async loadVideo() {
      this.error = '';
      this.videoUrl = '';
      this.loading = true;

      try {
        if (!this.videoInputUrl) {
          // 默认从后端获取示例 HLS URL
          const response = await axios.get('http://localhost:5000/api/video');
          this.videoUrl = response.data.video_url;
        } else if (this.videoInputUrl.startsWith('rtmp://')) {
          // 处理 RTMP URL
          const response = await axios.post('http://localhost:5000/api/video', {
            rtmp_url: this.videoInputUrl
          });
          this.videoUrl = response.data.video_url;
        } else {
          this.error = '请输入有效的 RTMP URL（必须以 rtmp:// 开头）';
        }
      } catch (error) {
        this.error = error.response?.data?.error || '无法加载视频，请检查 URL 或服务器状态';
        console.error('加载视频失败:', error);
      } finally {
        this.loading = false;
      }
    }
  }
};
</script>

<style>
input, button {
  font-size: 16px;
}
button:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
}
</style>