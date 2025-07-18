<template>
  <div class="data-entry-container">
    <h2 class="page-section-title">新增摄像头</h2>
    <div class="form-card">
      <form @submit.prevent="addCamera">
        <div class="form-group">
          <label for="cameraName">摄像头名称:</label>
          <input type="text" id="cameraName" v-model="camera.name" required placeholder="请输入摄像头名称">
        </div>

        <div class="form-group">
          <label for="cameraPlace">安装地点:</label>
          <input type="text" id="cameraPlace" v-model="camera.place" required placeholder="请输入安装地点">
        </div>

        <div class="form-group">
          <label for="cameraType">摄像头类型:</label>
          <select id="cameraType" v-model="camera.type" required>
            <option value="">请选择类型</option>
            <option value="face">人脸识别</option>
            <option value="danger">危险区域检测</option>
            <!-- 保持与后端接口文档一致的类型，如果需要更多类型，请确保后端也支持 -->
          </select>
        </div>

        <button type="submit" class="submit-button">添加摄像头</button>
      </form>
    </div>

    <!-- 移除 v-if="showCustomMessage"，让 CustomMessage 始终存在于 DOM 中 -->
    <!-- 通过 @close 事件清空 customMessageText，CustomMessage 会自动隐藏 -->
    <CustomMessage
      :message="customMessageText"
      :type="customMessageType"
      @close="customMessageText = ''" 
    />
  </div>
</template>

<script>
import axios from 'axios';
import CustomMessage from './CustomMessage.vue'; // 确保 CustomMessage 组件路径正确

export default {
  name: 'DataEntryPage',
  components: {
    CustomMessage,
  },
  data() {
    return {
      currentUserId: null, 
      // 移除 showCustomMessage 属性，不再需要它来控制 CustomMessage 的渲染
      // showCustomMessage: false, // <-- 移除此行

      camera: {
        userId: null, // 将在提交前设置为 currentUserId
        name: '',
        place: '',
        type: '',
      },
      customMessageType: 'info',
      customMessageText: '',
    };
  },
  created() {
    // 在组件创建时，从 localStorage 获取用户 ID
    const storedUserId = localStorage.getItem('user_id');
    if (storedUserId) {
      // 将获取到的字符串转换为数字类型（如果 userId 是数字）
      this.currentUserId = parseInt(storedUserId, 10);
      console.log('从 localStorage 获取到用户ID:', this.currentUserId);
    } else {
      console.warn('localStorage 中未找到用户ID。用户可能未登录或登录信息已过期。');
      // 可以在这里添加重定向到登录页的逻辑，例如：
      // this.$router.push({ name: 'LoginPage' });
      this.showMessage('请先登录以添加摄像头。', 'error');
    }
  },
  methods: {
    async addCamera() {
      // 检查用户ID是否已获取
      if (this.currentUserId === null) {
        this.showMessage('用户ID未获取，无法添加摄像头。请尝试重新登录。', 'error');
        return;
      }

      // 1. 设置用户ID
      this.camera.userId = this.currentUserId;

      // 2. 简单的前端验证
      if (!this.camera.name || !this.camera.place || !this.camera.type) {
        this.showMessage('请填写所有必填字段！', 'warning');
        return;
      }

      try {
        const response = await axios.post('/data/cameras', this.camera);

        if (response.status === 201) { // 后端返回 201 表示创建成功
          this.showMessage(`摄像头 "${this.camera.name}" 添加成功！ID: ${response.data.id || 'N/A'}`, 'success');
          // 成功后清空表单
          this.camera.name = '';
          this.camera.place = '';
          this.camera.type = '';
        } else {
          // 理论上 201 以外的状态会被 catch 块捕获，但以防万一
          this.showMessage('添加摄像头失败，请检查数据。', 'error');
        }
      } catch (error) {
        console.error('添加摄像头失败:', error);
        if (error.response) {
          // 根据后端返回的错误码和信息显示具体错误
          if (error.response.status === 400) {
            this.showMessage(`参数错误或摄像头名称已存在: ${error.response.data.message || '请检查输入。'}`, 'error');
          } else if (error.response.status === 404) {
            this.showMessage('用户不存在，请确认登录状态。', 'error');
          } else {
            this.showMessage(`添加摄像头失败: ${error.response.data.message || error.message || '未知错误'}`, 'error');
          }
        } else {
          this.showMessage('网络请求失败，请检查网络连接。', 'error');
        }
      }
    },
    showMessage(message, type) {
      this.customMessageText = message;
      this.customMessageType = type;
      // 移除 showCustomMessage = true; 因为 CustomMessage 会根据 messageText 自动显示
      // this.showCustomMessage = true; // <-- 移除此行
    },
  },
};
</script>

<style scoped>
.data-entry-container {
  padding: 20px;
  background-color: #f0f2f5;
  min-height: calc(100vh - 40px); /* 减去padding，确保背景填充整个区域 */
}

.page-section-title {
  color: #2c3e50;
  margin-bottom: 25px;
  text-align: center;
  font-size: 2em;
  font-weight: 700;
}

.form-card {
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  padding: 30px;
  max-width: 600px;
  margin: 0 auto; /* 居中显示 */
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: bold;
  color: #34495e;
}

.form-group input[type="text"],
.form-group select {
  width: calc(100% - 20px); /* 减去padding */
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
  font-size: 1em;
  box-sizing: border-box; /* 确保padding和border包含在宽度内 */
  transition: border-color 0.3s ease;
}

.form-group input[type="text"]:focus,
.form-group select:focus {
  border-color: #3498db; /* 聚焦时边框变色 */
  outline: none;
}

.submit-button {
  display: block;
  width: 100%;
  padding: 12px 20px;
  background-color: #3498db;
  color: white;
  border: none;
  border-radius: 5px;
  font-size: 1.1em;
  cursor: pointer;
  transition: background-color 0.3s ease;
  margin-top: 20px;
}

.submit-button:hover {
  background-color: #2980b9;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .form-card {
    padding: 20px;
  }
}
</style>
