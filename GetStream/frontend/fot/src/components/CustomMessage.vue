<template>
  <transition name="fade">
    <div v-if="visible" :class="['custom-message', type]">
      <p>{{ internalMessage }}</p>
    </div>
  </transition>
</template>

<script>
export default {
  name: 'CustomMessage',
  props: {
    message: { // 这是一个 prop，用于从父组件接收初始消息
      type: String,
      default: ''
    },
    type: {
      type: String,
      default: 'info', // 'success', 'error', 'info', 'warning'（虽然 warning 在样式上与 info 相同）
      validator: (value) => ['success', 'error', 'info', 'warning'].includes(value)
    },
    duration: {
      type: Number,
      default: 3000 // 消息显示时长（毫秒）
    }
  },
  data() {
    return {
      visible: false,
      timer: null,
      internalMessage: '' // 内部数据属性来管理显示的消息
    };
  },
  watch: {
    // 监听 prop 的变化，并更新内部数据
    message(newVal) {
      // **只有当新消息非空时才显示。**
      // 这现在是触发消息显示的主要方式。
      if (newVal) {
        this.internalMessage = newVal; // 更新内部消息
        this.show(); // 显示消息
      } else {
        // 如果 message prop 变为空（例如，父组件清空了它），则隐藏消息
        this.hide();
      }
    }
  },
  methods: {
    show() {
      this.visible = true;
      if (this.timer) {
        clearTimeout(this.timer);
      }
      this.timer = setTimeout(() => {
        this.hide();
      }, this.duration);
    },
    hide() {
      this.visible = false;
      // 当消息隐藏时，发出一个 'close' 事件。父组件可以使用它来设置 `v-if` 为 false。
      this.$emit('close');
    }
  },
  // **移除了 `created` 钩子中的 `this.show()` 调用。**
  // 消息现在只会在 `message` prop 明确接收到非空值时才显示。
  /*
  created() {
    if (this.message) {
      this.internalMessage = this.message;
      this.show();
    }
  },
  */
  beforeUnmount() {
    // 在组件销毁前清除计时器，以防止内存泄漏
    if (this.timer) {
      clearTimeout(this.timer);
    }
  }
};
</script>

<style scoped>
/* 样式保持不变 */
.custom-message {
  position: fixed;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  padding: 12px 20px;
  border-radius: 8px;
  color: white;
  font-weight: bold;
  font-size: 1em;
  z-index: 9999; /* 确保在最上层 */
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  min-width: 250px;
  text-align: center;
}

.custom-message.success {
  background-color: #28a745; /* 绿色成功 */
}

.custom-message.error {
  background-color: #dc3545; /* 红色错误 */
}

.custom-message.info, .custom-message.warning { /* 信息和警告都使用此样式 */
  background-color: #17a2b8; /* 蓝色信息/警告 */
}

/* 过渡效果 */
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.5s, transform 0.5s;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(-20px);
}
</style>