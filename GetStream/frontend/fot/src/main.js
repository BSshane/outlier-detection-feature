import { createApp } from 'vue';
import App from './App.vue';
import router from '../router';
import sliderVerify from 'vue3-slider-verify';
import '../node_modules/vue3-slider-verify/lib/style.css'; // 引入样式
import axios from 'axios'; // 引入 axios
//import { ElMessage } from 'element-plus'; // 引入 ElMessage 用于错误提示

const app = createApp(App);

// --- Axios 全局配置 ---

// 设置 Axios 的基础 URL
// 在开发环境中，这可能是您的本地后端地址，例如 'http://localhost:8080'
// 在生产环境中，它将是您的部署后的后端地址
// axios.defaults.baseURL = 'http://1.92.135.70:9080'; // **请将此替换为您的实际后端 API 地址**

// 添加请求拦截器
// 每次发送请求前，检查是否存在 JWT 令牌，并将其添加到 Authorization 头中
axios.interceptors.request.use(
  config => {
    const token = localStorage.getItem('jwt_token'); // 从 Local Storage 获取令牌
    if (token) {
      config.headers.Authorization = `Bearer ${token}`; // 添加 Bearer 令牌
    }
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// --- Vue 应用挂载 ---
app.use(router); // 使用路由
app.use(sliderVerify);
app.mount('#app');