// C:\Users\24534\Desktop\outlier-detection\GetStream\frontend\fot\router\index.js

import { createRouter, createWebHistory } from 'vue-router';

// 确保导入的组件名称和文件路径完全匹配
import LendInPage from '../src/components/LendInPage.vue';
import WindowTwo from '@/views/WindowTwo.vue';
import AlarmInfoPage from '../src/components/AlarmInfoPage.vue';

const routes = [
  {
    path: '/',
    name: 'LendInPage', // 使用与导入组件一致的 name
    component: LendInPage
  },
  {
    path: '/window2', //主界面
    name: 'WindowTwo',
    component: WindowTwo
  },
  {
    path: '/alarm-info', // 定义告警信息页面的路径
    name: 'AlarmInfoPage', // 给路由命名
    component: AlarmInfoPage
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;