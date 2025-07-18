<template>
  <div class="alarm-info-container">
    <div class="main-content">
      <h1 class="page-title">告警信息管理</h1>

      <div class="content-wrapper">
        <div class="video-section">
          <h2>媒体流显示</h2>
          <div class="media-player">
            <video v-if="currentMediaUrl && mediaType === 'video'" :src="currentMediaUrl" controls autoplay muted class="responsive-media"></video>
            <img v-else-if="currentMediaUrl && mediaType === 'image'" :src="currentMediaUrl" alt="告警图片" class="responsive-media">
            <div v-else class="media-placeholder">
              <p>暂无媒体文件，请从告警列表中选择。</p>
            </div>
          </div>
        </div>

        <div class="alarm-list-section">
          <h2>告警列表</h2>
          <div class="filter-section">
            <label for="alarm-type-filter">筛选类型:</label>
            <select id="alarm-type-filter" v-model="selectedFilterType" class="type-filter-dropdown">
              <option value="">所有类型</option>
              <option v-for="(label, type) in alarmTypeLabels" :key="type" :value="type">{{ label }}</option>
            </select>
          </div>
          <ul class="alarm-list">
            <li v-for="alarm in filteredAlarms" :key="alarm.id"
                :class="{ 'alarm-item': true, 'selected': selectedAlarm && selectedAlarm.id === alarm.id, 'processed': alarm.isProcessed }"
                @click="selectAlarm(alarm)">
              <div class="alarm-details">
                <span class="alarm-id">ID: {{ alarm.id }}</span>
                <span class="alarm-time">{{ formatTime(alarm.curTime) }}</span>
                <span :class="['alarm-type', alarm.type.replace(/\s/g, '-')]">{{ getAlarmTypeLabel(alarm.type) }}</span>
              </div>
              <button
                @click.stop="processAlarm(alarm.id)"
                :class="['process-button', { 'processed-button': alarm.isProcessed }]">
                {{ alarm.isProcessed ? '已处理' : '处理告警' }}
              </button>
            </li>
            <li v-if="filteredAlarms.length === 0" class="no-alarms">
              暂无告警信息或当前筛选条件下无告警。
            </li>
          </ul>
        </div>
      </div>

      <div class="detail-section">
        <h2>详细信息</h2>
        <div v-if="selectedAlarm" class="alarm-detail-card">
          <p><strong>告警ID:</strong> {{ selectedAlarm.id }}</p>
          <p><strong>发生时间:</strong> {{ formatTime(selectedAlarm.curTime) }}</p>
          <p><strong>告警类型:</strong> <span :class="['alarm-type', selectedAlarm.type.replace(/\s/g, '-')]">{{ getAlarmTypeLabel(selectedAlarm.type) }}</span></p>
          <p><strong>媒体URL:</strong> <a :href="selectedAlarm.videoURL" target="_blank">{{ selectedAlarm.videoURL || '无' }}</a></p>
          <p><strong>详细描述:</strong> {{ selectedAlarm.info || '无详细信息' }}</p>
          <p><strong>处理状态:</strong> <span :class="{'status-processed': selectedAlarm.isProcessed, 'status-pending': !selectedAlarm.isProcessed}">{{ selectedAlarm.isProcessed ? '已处理' : '待处理' }}</span></p>
        </div>
        <div v-else class="no-selection">
          <p>请从上方告警列表中选择一条告警信息以查看详细信息。</p>
        </div>
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
import CustomMessage from './CustomMessage.vue'; // 导入您的 CustomMessage 组件
import {alarmApi} from '../api/axios.js'; // 导入您的 alarmApi

export default {
  name: 'AlarmInfoPage',
  components: {
    CustomMessage // 注册 CustomMessage 组件
  },
  data() {
    return {
      alarms: [],
      selectedAlarm: null,
      currentMediaUrl: '',
      mediaType: '',
      // 用于 CustomMessage 的新数据属性
      showCustomMessage: false,
      customMessageType: 'info',
      customMessageText: '',
      // 新增：筛选告警类型的数据属性
      selectedFilterType: '', // 默认为空，表示不过滤，显示所有
      alarmTypeLabels: {
        'stranger': '陌生人',
        'cheat': '非真人',
        'helmet': '头盔',
        'dangerous area': '危险区域',
        'tumble': '摔倒'
      }
    };
  },
  props: {
    activeMenuItem: {
      type: String,
      default: 'alarmInfo'
    }
  },
  computed: {
    // 根据 selectedFilterType 筛选告警列表
    filteredAlarms() {
      if (!this.selectedFilterType) {
        return this.alarms; // 如果没有选择筛选类型，则返回所有告警
      }
      return this.alarms.filter(alarm => alarm.type === this.selectedFilterType);
    }
  },
  created() {
    this.fetchAlarms();
  },
  methods: {
    async fetchAlarms() {
      try {
        const response = await alarmApi.get('/data/warnings');
        this.alarms = response.data.map(alarm => ({
          ...alarm,
          // 如果后端没有提供处理状态，默认设置为 false
          isProcessed: alarm.isProcessed || false
        }));
        this.showMessage('告警信息加载成功！', 'success');
      } catch (error) {
        console.error('获取告警信息失败:', error);
        this.showMessage('获取告警信息失败，请稍后再试。', 'error');
      }
    },
    formatTime(isoString) {
      const date = new Date(isoString);
      return date.toLocaleString();
    },
    getAlarmTypeLabel(type) {
      // 使用 alarmTypeLabels 映射，如果不存在则直接返回类型本身
      return this.alarmTypeLabels[type] || type;
    },
    selectAlarm(alarm) {
      this.selectedAlarm = alarm;
      this.currentMediaUrl = ''; // 重置
      this.mediaType = '';      // 重置

      if (this.selectedAlarm && this.selectedAlarm.videoURL) {
        let url = this.selectedAlarm.videoURL.toLowerCase();

        // **处理后端返回的相对图片路径**
        if (!url.startsWith('http://') && !url.startsWith('https://') && !url.startsWith('rtmp://')) {
            url = url.replace(/\\/g, '/'); // 确保路径分隔符正确
            this.currentMediaUrl = `${axios.defaults.baseURL}/${url}`;
        } else {
            this.currentMediaUrl = this.selectedAlarm.videoURL;
        }

        // 简单的通过文件扩展名判断类型
        if (url.endsWith('.mp4') || url.endsWith('.webm') || url.endsWith('.avi') || url.endsWith('.mov') || url.endsWith('.flv')) {
          this.mediaType = 'video';
        } else if (url.endsWith('.jpg') || url.endsWith('.jpeg') || url.endsWith('.png') || url.endsWith('.gif') || url.endsWith('.bmp')) {
          this.mediaType = 'image';
        } else if (url.startsWith('rtmp://')) {
          // RTMP 流通常由特定播放器处理，HTML <video> 标签不直接支持
          this.mediaType = 'video';
          this.showMessage('检测到 RTMP 流，可能需要特定播放器或浏览器插件支持。', 'info');
        } else {
          this.showMessage('无法识别的媒体文件类型。', 'warning');
          this.mediaType = ''; // 未知类型
        }
      } else {
        this.showMessage('所选告警无媒体URL。', 'warning');
      }
    },
    async processAlarm(alarmId) {
      const alarmIndex = this.alarms.findIndex(a => a.id === alarmId);
      if (alarmIndex !== -1) {
        try {
          // **关键修改：切换 isProcessed 状态**
          const newStatus = !this.alarms[alarmIndex].isProcessed;

          // TODO: 这里应添加实际的后端 API 调用来更新告警状态
          // 例如：
          // await axios.put(`/api/warnings/${alarmId}/status`, { isProcessed: newStatus });

          // 更新本地状态
          this.alarms[alarmIndex].isProcessed = newStatus;

          // 如果当前选中项是正在处理的告警，同步更新其状态
          if (this.selectedAlarm && this.selectedAlarm.id === alarmId) {
            this.selectedAlarm.isProcessed = newStatus;
          }

          // 根据新状态显示不同的消息
          if (newStatus) {
            this.showMessage(`告警 ID: ${alarmId} 已处理！`, 'success');
          } else {
            this.showMessage(`告警 ID: ${alarmId} 已恢复为待处理！`, 'info');
          }

        } catch (error) {
          console.error(`处理告警 ID: ${alarmId} 失败:`, error);
          this.showMessage(`处理告警 ID: ${alarmId} 失败，请重试。`, 'error');
        }
      }
    },
    showMessage(message, type) {
      this.customMessageText = message;
      this.customMessageType = type;
      this.showCustomMessage = true; // 显示自定义消息组件
    }
  }
};
</script>

<style scoped>
/* 样式保持不变 */
.alarm-info-container {
  display: flex;
  min-height: 100vh;
  background-color: #f0f2f5; /* 整体浅灰色背景 */
}

/* 主内容区样式 */
.main-content {
  flex-grow: 1; /* 占据剩余空间 */
  padding: 20px 30px;
  display: flex;
  flex-direction: column;
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
  gap: 25px; /* 间距 */
  margin-bottom: 25px;
  flex-wrap: wrap; /* 允许换行，适应小屏幕 */
}

/* 媒体显示区 */
.video-section {
  flex: 3; /* 占据更多空间 */
  min-width: 400px;
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.video-section h2 {
  color: #34495e;
  margin-top: 0;
  margin-bottom: 15px;
  font-size: 1.5em;
}

.media-player {
  width: 100%;
  max-width: 640px; /* 限制媒体播放器最大宽度 */
  background-color: #000;
  border-radius: 6px;
  overflow: hidden;
  position: relative;
  padding-top: 56.25%; /* 16:9 宽高比 */
  margin-bottom: 20px;
}

.responsive-media {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain; /* 保持媒体比例，适应框内 */
}

.media-placeholder {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #333;
  color: #bbb;
  font-size: 1.1em;
}

.action-button {
  padding: 10px 20px;
  background-color: #3498db; /* 蓝色按钮 */
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1em;
  transition: background-color 0.3s ease;
}

.action-button:hover:not(:disabled) {
  background-color: #2980b9;
}

.action-button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

.load-video-button {
  margin-top: 15px;
  width: auto; /* 按钮宽度自适应 */
}

/* 告警列表区 */
.alarm-list-section {
  flex: 2; /* 占据较少空间 */
  min-width: 300px;
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  padding: 20px;
  overflow-y: auto; /* 允许滚动 */
  max-height: 500px; /* 限制最大高度 */
}

.alarm-list-section h2 {
  color: #34495e;
  margin-top: 0;
  margin-bottom: 15px;
  font-size: 1.5em;
}

.filter-section {
  margin-bottom: 15px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.filter-section label {
  font-weight: bold;
  color: #555;
}

.type-filter-dropdown {
  padding: 8px 12px;
  border: 1px solid #ccc;
  border-radius: 5px;
  background-color: #f9f9f9;
  font-size: 0.95em;
  color: #333;
  cursor: pointer;
  outline: none; /* 移除选中时的蓝色边框 */
  transition: border-color 0.2s ease;
}

.type-filter-dropdown:hover {
  border-color: #999;
}

.type-filter-dropdown:focus {
  border-color: #3498db;
  box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
}


.alarm-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.alarm-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 15px;
  border-bottom: 1px solid #eee;
  cursor: pointer;
  transition: background-color 0.2s ease, border-color 0.2s ease;
}

.alarm-item:last-child {
  border-bottom: none;
}

.alarm-item:hover {
  background-color: #f5f5f5;
}

.alarm-item.selected {
  background-color: #e0eaff; /* 选中项的背景色 */
  border-left: 5px solid #3498db;
  padding-left: 10px; /* 适应边框 */
}

.alarm-item.processed {
  opacity: 0.7; /* 已处理的告警可以稍微变灰 */
  font-style: italic;
}

.alarm-details {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  flex-grow: 1;
}

.alarm-id {
  font-weight: bold;
  color: #333;
}

.alarm-time {
  font-size: 0.9em;
  color: #666;
  margin-top: 3px;
}

.alarm-type {
  font-size: 0.85em;
  padding: 4px 8px;
  border-radius: 4px;
  margin-top: 5px;
  color: white;
  display: inline-block; /* 确保背景色和圆角正常显示 */
}

/* 告警类型颜色 */
.stranger { background-color: #e74c3c; } /* 红色 */
.cheat { background-color: #f39c12; } /* 橙色 */
.helmet { background-color: #2ecc71; } /* 绿色 */
.dangerous-area { background-color: #9b59b6; } /* 紫色 */
.tumble { background-color: #3498db; } /* 蓝色 */


.process-button {
  padding: 8px 12px;
  background-color: #28a745; /* 绿色处理按钮 */
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.85em;
  transition: background-color 0.3s ease;
  margin-left: 10px;
  white-space: nowrap; /* 防止按钮文字换行 */
}

.process-button:hover:not(:disabled) {
  background-color: #218838;
}

.processed-button {
  background-color: #6c757d; /* 已处理按钮颜色 */
  cursor: pointer; /* 保持指针，表示仍可点击 */
}

.no-alarms {
  text-align: center;
  padding: 20px;
  color: #999;
}


/* 详细信息区 */
.detail-section {
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  padding: 20px;
}

.detail-section h2 {
  color: #34495e;
  margin-top: 0;
  margin-bottom: 15px;
  font-size: 1.5em;
}

.alarm-detail-card p {
  margin-bottom: 8px;
  line-height: 1.6;
  color: #555;
}

.alarm-detail-card p strong {
  color: #333;
}

.status-processed {
  color: #28a745; /* 绿色 */
  font-weight: bold;
}

.status-pending {
  color: #dc3545; /* 红色 */
  font-weight: bold;
}

.no-selection {
  text-align: center;
  color: #999;
  padding: 20px;
}

/* 媒体查询 - 响应式布局 */
@media (max-width: 768px) {
  .alarm-info-container {
    flex-direction: column; /* 侧边栏和主内容垂直排列 */
  }

  .main-content {
    padding: 15px;
  }

  .content-wrapper {
    flex-direction: column; /* 视频和列表垂直堆叠 */
  }

  .video-section, .alarm-list-section {
    min-width: unset; /* 取消最小宽度限制 */
    width: 100%; /* 占据全宽 */
  }

  .alarm-list-section {
    max-height: 350px; /* 调整小屏幕下的列表最大高度 */
  }

  .alarm-item {
    flex-direction: column; /* 告警项内部垂直布局 */
    align-items: flex-start;
    gap: 8px;
  }

  .process-button {
    margin-left: 0;
    margin-top: 8px;
    width: 100%;
  }
}
</style>