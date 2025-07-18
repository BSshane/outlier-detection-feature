<template>
  <aside class="sidebar left-sidebar">
    <div class="sidebar-header">
      <h2>系统功能</h2>
    </div>
    <nav class="main-menu">
      <ul>
        <li @click="selectMenuItem('dataEntry')" :class="{ active: activeMenu === 'dataEntry' }">
          <i class="icon-entry"></i> 信息录入
        </li>
        <li @click="selectMenuItem('faceRecognition')" :class="{ active: activeMenu === 'faceRecognition' }">
          <i class="icon-face"></i> 人脸识别
        </li>
        
        <li @click="selectMenuItem('fallDetection')" :class="{ active: activeMenu === 'fallDetection' }">
          <i class="icon-fall"></i> 跌倒检测
        </li>

        <li @click="selectMenuItem('audioDetection')" :class="{ active: activeMenu === 'audioDetection' }">
          <i class="icon-audio"></i> 音频检测
        </li>

        <li @click="selectMenuItem('alarmInfo')" :class="{ active: activeMenu === 'alarmInfo' }">
          <i class="icon-alarm"></i> 告警信息
        </li>
        <li @click="selectMenuItem('dailyReport')" :class="{ active: activeMenu === 'dailyReport' }">
          <i class="icon-report"></i> 监控日志 </li>
        <li @click="selectMenuItem('historyQuery')" :class="{ active: activeMenu === 'historyQuery' }">
          <i class="icon-history"></i> 目标检测
        </li>
      </ul>
    </nav>
  </aside>
</template>

<script>
export default {
  name: 'DashboardSidebar',
  props: {
    activeMenu: {
      type: String,
      default: 'dataEntry'
    }
  },
  methods: {
    selectMenuItem(menuItem) {
      this.$emit('menu-selected', menuItem);
    }
  }
};
</script>

<style scoped>
/* 侧边栏基础样式 */
.sidebar {
  width: 250px;
  background-color: #2c3e50; /* 深色背景 */
  color: #ecf0f1;
  padding: 20px 0;
  box-shadow: 2px 0 6px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
}

.left-sidebar {
  border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* 侧边栏头部 */
.sidebar-header {
  padding: 0 20px 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  margin-bottom: 20px;
}
.sidebar-header h2 {
  margin: 0;
  color: #f39c12; /* 强调色 */
}

/* 主菜单 */
.main-menu ul {
  list-style: none;
  padding: 0;
  margin: 0;
}
.main-menu li {
  padding: 15px 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  transition: background-color 0.3s ease, color 0.3s ease;
}
.main-menu li:hover {
  background-color: #34495e; /* 悬停背景 */
}
.main-menu li.active {
  background-color: #f39c12; /* 选中背景色 */
  color: #2c3e50; /* 选中文字颜色 */
  font-weight: bold;
}
.main-menu li i {
  /* 占位符图标样式 */
  width: 20px;
  height: 20px;
  background-color: rgba(255, 255, 255, 0.3);
  border-radius: 3px;
}
/* 实际应用中，这里会用字体图标或SVG */
.icon-entry::before { content: 'i'; }
.icon-face::before { content: '🧑'; }
.icon-alarm::before { content: '🔔'; }
.icon-report::before { content: '📄'; }
.icon-history::before { content: '⏰'; }
.icon-audio::before { content: '🔊'; }


/* 侧边栏底部 */
.sidebar-footer {
  margin-top: auto; /* 推到最底部 */
  padding: 20px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

/* 小屏幕适配 */
@media (max-width: 1200px) {
  .sidebar {
    width: 100%; /* 侧边栏变宽 */
    padding: 15px;
    box-shadow: none;
    border-right: none;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  }
  .left-sidebar {
    order: -1; /* 左侧栏优先显示 */
  }
  .sidebar-header {
    text-align: center;
  }
  .main-menu ul {
    display: flex; /* 菜单项横向排列 */
    flex-wrap: wrap;
    justify-content: center;
  }
  .main-menu li {
    flex-basis: 30%; /* 每个菜单项占据的宽度 */
    text-align: center;
    justify-content: center;
    padding: 10px;
    margin: 5px;
  }
  .sidebar-footer {
    padding: 15px;
    text-align: center;
  }
}
</style>