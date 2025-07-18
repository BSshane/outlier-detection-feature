<template>
  <div class="login-container">
    <div class="login-box">
      <h1 class="login-title">
        <span v-if="isLogin">登录您的账户</span>
        <span v-else>注册新账户</span>
        <span class="highlight"></span>
      </h1>
      <form @submit.prevent="handleLogin" v-if="isLogin">
        <div class="form-group">
          <label for="username" class="input-label">用户名</label>
          <input type="text" id="username" v-model="username" required class="input-field">
        </div>
        <div class="form-group">
          <label for="password" class="input-label">密码</label>
          <input type="password" id="password" v-model="password" required class="input-field">
        </div>
        <button type="submit" class="login-button">登录系统</button>
      </form>

      <form @submit.prevent="handleRegister" v-else>
        <div class="form-group">
          <label for="regUsername" class="input-label">用户名</label>
          <input type="text" id="regUsername" v-model="regUsername" required class="input-field">
        </div>
        <div class="form-group">
          <label for="regPassword" class="input-label">密码</label>
          <input type="password" id="regPassword" v-model="regPassword" required class="input-field">
        </div>
        <div class="form-group">
          <label for="confirmPassword" class="input-label">确认密码</label>
          <input type="password" id="confirmPassword" v-model="confirmPassword" required class="input-field">
        </div>
        <button type="submit" class="login-button">立即注册</button>
      </form>

      <p class="toggle-form-text">
        <span v-if="isLogin">没有账户？<a href="#" @click.prevent="toggleForm" class="toggle-link">立即注册</a></span>
        <span v-else>已有账户？<a href="#" @click.prevent="toggleForm" class="toggle-link">返回登录</a></span>
      </p>

      <p class="footer-text">
        智慧管理 <span class="highlight">从这里开始</span>
      </p>
    </div>

    <slider-verify-dialog
      v-if="loginAttempted"
      ref="sliderDialog"
      @onSuccess="handleVerifySuccess"
      @onError="handleVerifyError"
      @onClose="handleVerifyClose"
    ></slider-verify-dialog>

    <CustomMessage
      :message="messageText"
      :type="messageType"
      @close="messageText = ''"
      v-if="messageText"
    ></CustomMessage>
  </div>
</template>

<script>
import SliderVerifyDialog from './SliderVerifyDialog.vue'; // 确保路径正确
import axios from 'axios'; // 引入 axios
import CustomMessage from './CustomMessage.vue'; // 引入自定义消息组件
import { alarmApi } from '@/api/axios.js';

export default {
  name: 'LoginPage',
  components: {
    SliderVerifyDialog,
    CustomMessage
  },
  data() {
    return {
      isLogin: true,
      username: '',
      password: '',
      regUsername: '',
      regPassword: '',
      confirmPassword: '',
      loginAttempted: false,
      messageText: '',
      messageType: 'info'
    };
  },
  methods: {
    handleLogin() {
      if (!this.username || !this.password) {
        this.showMessage('请输入用户名和密码！', 'error');
        return;
      }
      this.loginAttempted = true;
      // 您可能需要在 SliderVerifyDialog 上添加一个 open 方法，并在此时调用
      this.$nextTick(() => {
        if (this.$refs.sliderDialog && typeof this.$refs.sliderDialog.openDialog === 'function') {
          this.$refs.sliderDialog.openDialog();
        }
      });
    },
    async handleRegister() {
      if (!this.regUsername || !this.regPassword || !this.confirmPassword) {
        this.showMessage('请填写所有注册信息！', 'error');
        return;
      }
      if (this.regPassword !== this.confirmPassword) {
        this.showMessage('两次输入的密码不一致！', 'error');
        return;
      }
      if (this.regPassword.length < 6) {
        this.showMessage('密码长度至少为6位！', 'error');
        return;
      }

      try {
        const response = await alarmApi.post('/auth/register', {
          username: this.regUsername,
          password: this.regPassword,
          authority: 'admin'
        });

        if (response.status === 201) {
          this.showMessage('注册成功！请登录。', 'success');
          this.toggleForm();
          this.resetRegisterForm();
        }
      } catch (error) {
        console.error('注册失败:', error);
        if (error.response && error.response.status === 400) {
          this.showMessage('用户名已存在或参数错误！', 'error');
        } else {
          this.showMessage('注册失败，请稍后再试！', 'error');
        }
      }
    },
    toggleForm() {
      this.isLogin = !this.isLogin;
      this.resetForms();
      this.messageText = '';
    },
    resetForms() {
      this.username = '';
      this.password = '';
      this.regUsername = '';
      this.regPassword = '';
      this.confirmPassword = '';
    },
    resetRegisterForm() {
      this.regUsername = '';
      this.regPassword = '';
      this.confirmPassword = '';
    },
    async handleVerifySuccess(verifyResult) {
      console.log('滑动验证成功:', verifyResult);
      if (this.loginAttempted) {
        try {
          const response = await alarmApi.post('/auth/login', {
            username: this.username,
            password: this.password
          });

          if (response.status === 200) {
            this.showMessage('登录成功！', 'success');
            // 存储 JWT token
            localStorage.setItem('jwt_token', response.data.access_token); // 注意：这里是 access_token

            // **核心修改：从 response.data.user.id 获取 userId**
            if (response.data.user && response.data.user.id !== undefined && response.data.user.id !== null) {
              localStorage.setItem('user_id', response.data.user.id); // 存储用户ID
              console.log('用户ID已存储到 localStorage:', response.data.user.id);
            } else {
              console.warn('后端登录响应中未包含 user.id 字段，请检查API接口。');
            }

            this.$router.push({ name: 'WindowTwo' });
          }
        } catch (error) {
          console.error('登录失败:', error);
          if (error.response) {
            if (error.response.status === 401) {
              this.showMessage('用户名或密码错误！', 'error');
            } else {
              this.showMessage(`登录失败: ${error.response.data.message || error.message || '未知错误'}`, 'error');
            }
          } else {
            this.showMessage('网络请求失败，请检查网络连接。', 'error');
          }
        }
      }
      this.loginAttempted = false;
    },
    handleVerifyError(verifyResult) {
      console.log('滑动验证失败:', verifyResult);
      this.showMessage(verifyResult?.message || '验证失败，请重试！', 'error');
      this.loginAttempted = false;
    },
    handleVerifyClose() {
      console.log('滑动验证弹窗已关闭');
      this.loginAttempted = false;
      this.showMessage('', 'info');
    },
    // 修改 showMessage 方法以使用自定义组件
    showMessage(message, type) {
      this.messageText = message;
      this.messageType = type;
      // CustomMessage 组件会监听 messageText 变化自动显示
    }
  }
};
</script>

<style scoped>
/* 保持原有的样式 */
/* 整体容器 */
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  /* 背景改为图片，使用相对路径 */
  background: url('../assets/OIP.webp') no-repeat center center fixed;
  background-size: cover; /* 确保图片覆盖整个容器 */
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #ecf0f1; /* 整体文字颜色改为亮色 */
  overflow: hidden;
  position: relative;
}

/* 移除背景动画效果，因为现在是图片背景 */
.login-container::before {
  content: none; /* 移除伪元素，不再需要背景动画 */
}

@keyframes backgroundPulse {
  0% { transform: scale(1); opacity: 0.8; }
  100% { transform: scale(1.1); opacity: 1; }
}

/* 登录框 */
.login-box {
  background: rgba(72, 0, 0, 0.9); /* 更深的半透明背景 */
  padding: 40px;
  border-radius: 12px;
  /* 调整阴影颜色 */
  box-shadow: 0 10px 40px 0 rgb(0, 0, 0); /* 更强的阴影 */
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.324); /* 轻微的白色边框 */
  text-align: center;
  width: 100%;
  max-width: 420px;
  position: relative;
  z-index: 10;
  overflow: hidden;
}

/* 登录框炫光效果 */
.login-box::after {
  content: '';
  position: absolute;
  top: -5px;
  left: -5px;
  right: -5px;
  bottom: -5px;
  /* 炫光颜色改为橙色/琥珀色，模拟工地安全色 */
  background: linear-gradient(45deg, #fefefd, #fe8215, #ffffff);
  background-size: 400% 400%;
  filter: blur(20px);
  border-radius: 17px;
  z-index: -1;
  animation: glowing 5s linear infinite;
  opacity: 0.7; /* 调整不透明度 */
}

@keyframes glowing {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* 标题 */
.login-title {
  color: #ecf0f1; /* 亮白色，在高对比度下更清晰 */
  margin-bottom: 40px;
  font-size: 2.5em;
  font-weight: 800;
  letter-spacing: 2px;
  text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.4); /* 增加阴影深度 */
}

.highlight {
  color: #f39c12; /* 安全橙色作为强调色 */
  font-weight: bold;
}

/* 表单组 */
.form-group {
  margin-bottom: 25px;
  text-align: left;
}

/* 输入框标签 */
.input-label {
  display: block;
  margin-bottom: 10px;
  font-weight: 500;
  color: #d0cccc; /* 浅灰色，在深色背景上可见 */
  font-size: 0.95em;
  letter-spacing: 0.5px;
}

/* 输入框 */
.input-field {
  width: 100%;
  padding: 12px 15px;
  background: rgba(255, 255, 255, 0.08); /* 极浅的透明白色背景 */
  border: 1px solid rgba(255, 255, 255, 0.2); /* 浅色边框 */
  border-radius: 6px;
  font-size: 1.1em;
  color: rgba(255, 255, 255, 0.607); /* 输入文本颜色为亮白色 */
  box-sizing: border-box;
  outline: none;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.input-field:focus {
  border-color: #f39c12; /* 聚焦时边框颜色为强调色 */
  box-shadow: 0 0 8px rgba(243, 156, 18, 0.6); /* 聚焦时阴影为强调色光晕 */
}

/* 登录按钮 */
.login-button {
  width: 100%;
  padding: 15px;
  font-size: 1.2em;
  cursor: pointer;
  /* 按钮背景渐变使用强调色 */
  background: linear-gradient(45deg, #f39c12, #e67e22); /* 橙色渐变 */
  color: white; /* 按钮文字为白色 */
  border: none;
  border-radius: 8px;
  margin-top: 30px;
  font-weight: 600;
  letter-spacing: 1px;
  /* 按钮阴影颜色与强调色一致 */
  box-shadow: 0 5px 15px rgba(243, 156, 18, 0.4);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.login-button:hover {
  background: linear-gradient(45deg, #e67e22, #d35400); /* 悬停时颜色更深 */
  box-shadow: 0 8px 20px rgba(243, 156, 18, 0.6);
  transform: translateY(-2px);
}

/* 按钮点击效果 */
.login-button:active {
  transform: translateY(0);
  box-shadow: 0 3px 10px rgba(243, 156, 18, 0.3);
}

/* 按钮内部发光效果 */
.login-button::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 300%;
  height: 300%;
  background: rgba(255, 255, 255, 0.15); /* 内部发光颜色为半透明白色 */
  border-radius: 50%;
  transition: all 0.75s ease-out;
  transform: translate(-50%, -50%) scale(0);
  opacity: 0;
}

.login-button:hover::before {
  transform: translate(-50%, -50%) scale(1);
  opacity: 1;
}

/* 用于切换表单的文本和链接样式 */
.toggle-form-text {
  margin-top: 20px;
  font-size: 0.9em;
  color: #bdc3c7; /* 浅灰色 */
}

.toggle-link {
  color: #f39c12; /* 强调色链接 */
  text-decoration: none;
  font-weight: 600;
  transition: color 0.3s ease;
}

.toggle-link:hover {
  color: #e67e22; /* 悬停时颜色更深 */
  text-decoration: underline;
}

/* 底部文字 */
.footer-text {
  margin-top: 40px;
  font-size: 0.85em;
  color: #d8d9d9; /* 浅灰色 */
  letter-spacing: 0.5px;
}
</style>
