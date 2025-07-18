<template>
  <div class="daily-report-container">
    <div class="report-sidebar">
      <h2>日志日期</h2>
      <div class="date-picker-section">
        <h3>选择日期:</h3>
        <DatePicker
          v-model:value="selectedDateCalendar"
          @change="handleCalendarDateChange"
          :disabled-date="disableFutureDates"
          :type="'date'"
          :placeholder="'选择日期'"
          format="YYYY-MM-DD"
          value-type="YYYY-MM-DD"
          :lang="lang"
          input-class="calendar-input"
          :append-to-body="false" ></DatePicker>
      </div>
      <h3>最近7天的日志:</h3>
      <ul class="date-list">
        <li
          v-for="date in availableDates"
          :key="date"
          :class="{ 'date-item': true, 'selected': selectedDate === date }"
          @click="selectDate(date)">
          {{ formatDateDisplay(date) }}
        </li>
        <li v-if="availableDates.length === 0" class="no-dates">
          当前无可用日志日期。
        </li>
      </ul>
    </div>
    <div class="report-content">
      <div class="report-header">
        <h2 v-if="selectedDate">
          {{ formatDateDisplay(selectedDate) }} 监控日志
          <span v-if="loadingContent" class="loading-spinner"></span>
        </h2>
        <button
          v-if="selectedDate && !isGenerating && !loadingContent"
          @click="handleActionButton"
          :class="['action-button', logContent ? 'update' : 'generate']">
          {{ actionButtonText }}
        </button>
      </div>

      <div v-if="isGenerating" class="generation-status">
        <h3>请等待一分钟...</h3>
        <p>正在{{ actionButtonText }}，请稍候。</p>
        <div class="loading-spinner-large"></div>
        </div>

      <div v-else-if="loadingContent" class="loading-message">
        <p>正在加载日志内容...</p>
      </div>
      <div v-else-if="logContent" class="markdown-viewer" v-html="renderedMarkdown"></div>
      <div v-else class="no-content-selected">
        <p v-if="selectedDate">无法加载 {{ formatDateDisplay(selectedDate) }} 的日志，请检查文件是否存在。</p>
        <p v-else>请从左侧选择一个日期以查看监控日志。</p>
      </div>
    </div>

    <CustomMessage
      v-if="showCustomMessage"
      :message="customMessageText"
      :type="customMessageType"
      @close="showCustomMessage = false"
    />
  </div>
</template>

<script>
import axios from 'axios';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import CustomMessage from './CustomMessage.vue';

// 导入 DatePicker 组件和中文语言包
import DatePicker from 'vue-datepicker-next';
import 'vue-datepicker-next/index.css'; // 默认样式
import zh from 'vue-datepicker-next/locale/zh-cn'; // 中文语言包

export default {
  name: 'DailyReportPage',
  components: {
    CustomMessage,
    DatePicker // 注册 DatePicker 组件
  },
  data() {
    return {
      selectedMonth: '', // 这个现在主要用来初始化日历组件的显示月份
      availableDates: [], // 仍然用于显示过去7天的日期列表
      selectedDate: '', // 当前选中的日期，用于内容显示和后端请求
      selectedDateCalendar: null, // 用于DatePicker组件绑定的值 (YYYY-MM-DD 字符串)
      logContent: null,
      loadingContent: false,
      isGenerating: false,
      generationTimer: null,

      showCustomMessage: false,
      customMessageType: 'info',
      customMessageText: '',
      backendBaseUrl: 'http://127.0.0.1:9082/reports',
      
      lang: zh, // 设置日历语言为中文
    };
  },
  computed: {
    renderedMarkdown() {
      if (this.logContent) {
        const html = marked.parse(this.logContent);
        return DOMPurify.sanitize(html);
      }
      return '';
    },
    actionButtonText() {
      return this.logContent ? '更新日志' : '生成日志';
    },
  },
  created() {
    const today = new Date();
    const currentYear = today.getFullYear();
    const currentMonth = (today.getMonth() + 1).toString().padStart(2, '0');
    const currentDay = today.getDate().toString().padStart(2, '0');

    this.selectedMonth = `${currentYear}-${currentMonth}`; // 初始化月份选择器，但不直接用它生成日期列表
    this.generateDatesForMonth(); // 生成过去7天的日期列表

    // 尝试默认选中今天，并同步到日历组件
    const todayStr = `${currentYear}-${currentMonth}-${currentDay}`;
    this.selectDate(todayStr); // 尝试加载今天的数据
    this.selectedDateCalendar = todayStr; // 将日历组件的值也设为今天
  },
  methods: {
    generateDatesForMonth() {
      this.availableDates = [];
      const today = new Date();
      
      // 生成从今天开始，往回倒数7天的日期
      for (let i = 0; i < 7; i++) {
        const date = new Date(today);
        date.setDate(today.getDate() - i);
        
        const year = date.getFullYear();
        const month = (date.getMonth() + 1).toString().padStart(2, '0');
        const day = date.getDate().toString().padStart(2, '0');
        const dateStr = `${year}-${month}-${day}`;
        this.availableDates.push(dateStr);
      }
      this.availableDates.sort((a, b) => new Date(b) - new Date(a)); 

      // 确保在切换月份时，任何正在进行的生成操作的计时器被清除
      if (this.generationTimer) {
        clearTimeout(this.generationTimer);
        this.generationTimer = null;
        this.isGenerating = false; // 重置生成状态
      }
    },
    formatDateDisplay(dateStr) {
      const date = new Date(dateStr);
      // Ensure the date is valid before formatting
      if (isNaN(date.getTime())) {
        return "无效日期";
      }
      return date.toLocaleDateString('zh-CN', { year: 'numeric', month: 'long', day: 'numeric', weekday: 'short' });
    },
    
    // 处理日历组件选择日期
    handleCalendarDateChange(dateStr) {
      if (dateStr) {
        this.selectDate(dateStr);
      } else {
        // 用户可能清空了日期，此时可以重置一些状态
        this.selectedDate = '';
        this.logContent = null;
        this.loadingContent = false;
        this.isGenerating = false;
        if (this.generationTimer) {
          clearTimeout(this.generationTimer);
          this.generationTimer = null;
        }
      }
    },

    // 禁用未来日期的函数
    disableFutureDates(date) {
      const today = new Date();
      // 设置为当天的23:59:59，确保当天可选
      today.setHours(23, 59, 59, 999); 
      return date.getTime() > today.getTime();
    },

    async selectDate(date) {
      this.selectedDate = date;
      this.loadingContent = true;
      this.logContent = null; 
      
      if (this.generationTimer) {
        clearTimeout(this.generationTimer);
        this.generationTimer = null;
        this.isGenerating = false;
      }

      try {
        const response = await axios.get(`${this.backendBaseUrl}/get_report`, {
          params: { date: date }
        });

        if (response.data.status === 'success' && response.data.content) {
          this.logContent = response.data.content;
          this.showMessage(`已加载 ${this.formatDateDisplay(date)} 的日志。`, 'success');
        } else {
          this.logContent = null;
          this.showMessage(`未找到 ${this.formatDateDisplay(date)} 的日志。`, 'warning');
        }
      } catch (error) {
        console.error(`获取日志文件 ${date} 失败:`, error);
        this.logContent = null;
        if (error.response && error.response.status === 404) {
          this.showMessage(`未找到 ${this.formatDateDisplay(date)} 的日志文件。`, 'warning');
        } else {
          this.showMessage(`加载日志失败: ${error.message || '网络错误'}`, 'error');
        }
      } finally {
        this.loadingContent = false;
      }
    },
    handleActionButton() {
      if (!this.selectedDate) {
        this.showMessage('请先选择一个日期。', 'warning');
        return;
      }
      this.startGenerationProcess();
    },
    async startGenerationProcess() {
      this.isGenerating = true;
      this.logContent = null;
      this.showMessage(`正在${this.actionButtonText}日志，请等待一分钟...`, 'info');

      this.generationTimer = setTimeout(() => {
        // This timeout is just for the "please wait" message,
        // actual status is determined by the backend response.
      }, 60 * 1000); 

      try {
        const response = await axios.get(`${this.backendBaseUrl}/generate_report`, {
          params: { date: this.selectedDate }
        });

        if (response.data.status === 'success' && response.data.content) {
          this.logContent = response.data.content;
          this.showMessage(`${this.formatDateDisplay(this.selectedDate)} 的日志已${this.actionButtonText}成功！`, 'success');
        } else {
          this.showMessage(`${this.actionButtonText}失败: ${response.data.message || '未知错误'}`, 'error');
        }
      } catch (error) {
        console.error(`${this.actionButtonText}日志失败:`, error);
        this.showMessage(`${this.actionButtonText}失败: ${error.message || '网络错误'}`, 'error');
      } finally {
        this.isGenerating = false;
        if (this.generationTimer) {
          clearTimeout(this.generationTimer);
          this.generationTimer = null;
        }
      }
    },
    showMessage(message, type) {
      this.customMessageText = message;
      this.customMessageType = type;
      this.showCustomMessage = true;
    }
  },
  beforeUnmount() {
    if (this.generationTimer) {
      clearTimeout(this.generationTimer);
      this.generationTimer = null;
    }
  }
};
</script>

<style scoped>
/* 样式与您提供的一致，这里省略以保持简洁，但请确保在实际文件中包含它们 */
.daily-report-container {
  display: flex;
  min-height: calc(100vh - 40px); /* 减去父组件可能有的padding */
  background-color: #f8f9fa;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  margin: 20px; /* 给主窗口内容区留白 */
}

/* 左侧日志日期侧边栏 */
.report-sidebar {
  width: 300px;
  background-color: #ffffff;
  padding: 20px;
  border-right: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
  flex-shrink: 0; /* 防止缩小 */
}

.report-sidebar h2 {
  color: #34495e;
  margin-top: 0;
  margin-bottom: 20px;
  font-size: 1.6em;
  text-align: center;
}

.date-picker-section {
  margin-bottom: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

/* 调整选择日期 h3 的样式 */
.date-picker-section h3 {
  font-weight: bold;
  color: #34495e; /* 更深的颜色 */
  font-size: 1.2em; /* 增大字体 */
  margin-top: 10px;
  margin-bottom: 10px; /* 增加与日历的间距 */
  padding-bottom: 5px;
  border-bottom: 1px solid #e0e0e0; /* 添加一个细线 */
  text-align: left; /* 左对齐 */
}

/* 针对 vue-datepicker-next 的输入框样式 */
.date-picker-section .calendar-input {
  padding: 10px 15px; /* 增加内边距 */
  border: 1px solid #a0a0a0; /* 更明显的边框 */
  border-radius: 8px; /* 更圆润的边角 */
  font-size: 1.1em; /* 增大文字 */
  color: #333;
  cursor: pointer;
  outline: none;
  background-color: #f0f4f7; /* 浅灰色背景 */
  transition: border-color 0.3s ease, box-shadow 0.3s ease, background-color 0.3s ease;
  width: 100%; /* 确保输入框填充宽度 */
  box-sizing: border-box; /* 包含padding和border在宽度内 */
}

.date-picker-section .calendar-input:hover {
  border-color: #66b1ff; /* 鼠标悬停时蓝色边框 */
  background-color: #e6f7ff; /* 浅蓝色背景 */
}

.date-picker-section .calendar-input:focus {
  border-color: #3498db;
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.4); /* 更明显的阴影 */
  background-color: #ffffff; /* 选中时白色背景 */
}

/* 确保 DatePicker 的弹出层正确显示在侧边栏内部 */
/* 您可能需要根据实际布局微调此部分，如果 append-to-body="false" 不够 */
.mx-datepicker {
  width: 100%; /* 让 datepicker 组件占据其父容器的全部宽度 */
}

/* 最近7天的日志标题样式 */
.report-sidebar > h3 {
  color: #34495e;
  font-size: 1.2em; /* 统一标题字体大小 */
  margin-top: 20px; /* 与上方日历区域分隔 */
  margin-bottom: 10px;
  padding-bottom: 5px;
  border-bottom: 1px solid #e0e0e0;
  text-align: left;
}

.date-list {
  list-style: none;
  padding: 0;
  margin: 0;
  flex-grow: 1; /* 占据剩余空间 */
  overflow-y: auto; /* 日期列表可滚动 */
}

.date-item {
  padding: 12px 15px;
  cursor: pointer;
  border-bottom: 1px solid #f0f0f0;
  transition: background-color 0.2s ease, color 0.2s ease;
  color: #444;
  font-size: 1em;
}

.date-item:last-child {
  border-bottom: none;
}

.date-item:hover {
  background-color: #e6f7ff; /* 浅蓝色悬停效果 */
  color: #1890ff;
}

.date-item.selected {
  background-color: #1890ff; /* 选中项的蓝色背景 */
  color: #ffffff;
  font-weight: bold;
  border-left: 5px solid #096dd9;
  padding-left: 10px;
}

.no-dates {
  text-align: center;
  padding: 20px;
  color: #999;
  font-style: italic;
}

/* 右侧日志内容区 */
.report-content {
  flex-grow: 1; /* 占据剩余所有空间 */
  padding: 30px;
  background-color: #ffffff;
  overflow-y: auto; /* 日志内容可滚动 */
  color: #333;
  display: flex; /* 让内容垂直排列 */
  flex-direction: column;
}

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 25px;
  border-bottom: 1px solid #eee;
  padding-bottom: 10px;
}

.report-content h2 {
  color: #34495e;
  margin: 0; /* 移除默认外边距 */
  font-size: 1.8em;
  display: flex;
  align-items: center;
  gap: 10px;
}

.action-button {
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1em;
  font-weight: bold;
  transition: background-color 0.3s ease, transform 0.2s ease;
  white-space: nowrap; /* 防止文字换行 */
}

.action-button.generate {
  background-color: #28a745; /* 绿色 */
  color: white;
}

.action-button.generate:hover {
  background-color: #218838;
  transform: translateY(-1px);
}

.action-button.update {
  background-color: #007bff; /* 蓝色 */
  color: white;
}

.action-button.update:hover {
  background-color: #0056b3;
  transform: translateY(-1px);
}

.loading-message, .generation-status, .no-content-selected {
  text-align: center;
  padding: 50px;
  color: #666;
  font-size: 1.1em;
  flex-grow: 1; /* 这些状态容器应该填充可用空间 */
  display: flex;
  flex-direction: column;
  justify-content: center; /* 垂直居中 */
  align-items: center; /* 水平居中 */
}

.loading-spinner {
  border: 4px solid #f3f3f3; /* Light grey */
  border-top: 4px solid #3498db; /* Blue */
  border-radius: 50%;
  width: 20px;
  height: 20px;
  animation: spin 1s linear infinite;
  display: inline-block;
  vertical-align: middle;
}

.loading-spinner-large {
  border: 6px solid #f3f3f3; /* Light grey */
  border-top: 6px solid #3498db; /* Blue */
  border-radius: 50%;
  width: 60px;
  height: 60px;
  animation: spin 1.5s linear infinite;
  margin-top: 20px;
  margin-bottom: 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.markdown-viewer {
  line-height: 1.8;
  font-size: 1em;
  flex-grow: 1; /* Markdown 内容也占据剩余空间 */
}

/* 基本 Markdown 渲染样式，可以根据需要进一步美化 */
.markdown-viewer h1, .markdown-viewer h2, .markdown-viewer h3, .markdown-viewer h4 {
  color: #2c3e50;
  margin-top: 1.5em;
  margin-bottom: 0.8em;
}
.markdown-viewer p {
  margin-bottom: 1em;
}
.markdown-viewer ul, .markdown-viewer ol {
  margin-left: 20px;
  margin-bottom: 1em;
}
.markdown-viewer code {
  background-color: #f0f0f0;
  padding: 2px 4px;
  border-radius: 4px;
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
}
.markdown-viewer pre {
  background-color: #2d2d2d;
  color: #f8f8f2;
  padding: 15px;
  border-radius: 5px;
  overflow-x: auto;
  margin-bottom: 1em;
}
.markdown-viewer pre code {
  background-color: transparent;
  padding: 0;
  border-radius: 0;
  color: inherit;
}
.markdown-viewer table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 1em;
}
.markdown-viewer th, .markdown-viewer td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left;
}
.markdown-viewer th {
  background-color: #f2f2f2;
  font-weight: bold;
}
.markdown-viewer blockquote {
  border-left: 4px solid #ccc;
  padding-left: 10px;
  color: #666;
  margin: 1em 0;
}
.markdown-viewer a {
  color: #3498db;
  text-decoration: none;
}
.markdown-viewer a:hover {
  text-decoration: underline;
}


/* 响应式布局 */
@media (max-width: 768px) {
  .daily-report-container {
    flex-direction: column;
    margin: 10px;
  }
  .report-sidebar {
    width: 100%;
    border-right: none;
    border-bottom: 1px solid #e0e0e0;
    max-height: 300px;
  }
  .report-content {
    padding: 20px;
  }
  .date-item {
    padding: 10px 12px;
  }
  .report-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 15px;
  }
  .action-button {
    width: 100%;
  }
}
</style>