<template>
  <div class="window-two-layout">
    <DashboardSidebar @menu-selected="handleMenuSelected" :activeMenu="currentActiveMenu"></DashboardSidebar>
    <div class="content-area">
      <AlarmInfoPage v-if="currentActiveMenu === 'alarmInfo'"></AlarmInfoPage>
      <FaceRecognitionPage v-else-if="currentActiveMenu === 'faceRecognition'"></FaceRecognitionPage>
      <DataEntryPage v-else-if="currentActiveMenu === 'dataEntry'"></DataEntryPage>
      <FallDetectionPage v-else-if="currentActiveMenu === 'fallDetection'"></FallDetectionPage>
      <DailyReportPage v-else-if="currentActiveMenu === 'dailyReport'"></DailyReportPage> 
      <AudioDetectionPage v-else-if="currentActiveMenu === 'audioDetection'"></AudioDetectionPage>
      <HelmetMonitor v-else-if="currentActiveMenu === 'historyQuery'"></HelmetMonitor>
      <div v-else>
        <h3>欢迎来到主页</h3>
        <p>请从左侧菜单选择一个功能。</p>
      </div>
    </div>
  </div>
</template>

<script>
import DashboardSidebar from '../components/DashboardSidebar.vue';
import FaceRecognitionPage from "../components/FaceRecogContent.vue";
import AlarmInfoPage from '../components/AlarmInfoPage.vue';
import FallDetectionPage from '../components/FallDetectContent.vue'
import DataEntryPage from '../components/DataEntryPage.vue';
import DailyReportPage from '../components/DailyReportPage.vue'; // 引入新的监控日志页面
import AudioDetectionPage from '../components/AudioDetectionPage.vue'; 
import HelmetMonitor from '@/components/HelmetMonitor.vue';

export default {
  name: 'WindowTwo',
  components: {
    DashboardSidebar,
    FaceRecognitionPage,
    AlarmInfoPage,
    FallDetectionPage,
    DataEntryPage,
    DailyReportPage,
    AudioDetectionPage,
    HelmetMonitor,
  },
  data() {
    return {
      currentActiveMenu: 'dailyReport' // 默认显示监控日志页面，方便测试
    };
  },
  methods: {
    handleMenuSelected(menuItem) {
      this.currentActiveMenu = menuItem;
    }
  }
};
</script>

<style scoped>
.window-two-layout {
  display: flex;
  min-height: 100vh;
}
.content-area {
  flex-grow: 1;
  padding: 0; /* 这里的padding可以调整，或者让子组件自己控制 */
  background-color: #f0f2f5;
}
</style>