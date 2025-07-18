import axios from 'axios';

// 实例1：连接到告警信息后端 (url1)
const alarmApi = axios.create({
  baseURL: 'http://1.92.135.70:9080',
  timeout: 10000, // 请求超时时间 10秒
  // 还可以在这里为这个实例设置特定的 headers
  // headers: {'X-Custom-Header': 'AlarmService'}
});

// 实例2：连接到人脸识别后端 (url2)
// 假设这是你本地运行的 Flask 服务
const faceRecognitionApi = axios.create({
  baseURL: 'http://localhost:9083/api', // 你的 url2
  timeout: 25000, // 可以设置更长的超时时间，因为 AI 处理可能更慢
  // headers: {'X-Custom-Header': 'FaceRecognitionService'}
});

// 实例3：连接到音频监控后端 (新增)
const audioMonitorApi = axios.create({
  baseURL: 'http://127.0.0.1:9084', 
  timeout: 10000, // 请求超时时间
  headers: {
    'Content-Type': 'application/json',
  },
});

// 为所有需要认证的API实例添加请求拦截器
// 每次发送请求前，检查是否存在 JWT 令牌，并将其添加到 Authorization 头中
const addAuthInterceptor = (apiInstance) => {
  apiInstance.interceptors.request.use(
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
};

// 对需要认证的API实例应用拦截器
addAuthInterceptor(alarmApi);
addAuthInterceptor(faceRecognitionApi);
addAuthInterceptor(audioMonitorApi); // 对音频监控API也应用拦截器

// 导出所有这些实例，以便在其他组件中使用
export { alarmApi, faceRecognitionApi, audioMonitorApi };