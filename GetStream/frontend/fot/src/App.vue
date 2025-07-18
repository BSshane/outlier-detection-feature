<script setup>
// 这里不再需要导入HelloWorld，因为router-view会根据路由动态加载组件
</script>

<template>
  <div>
    <router-view></router-view>
  </div>
</template>

<style scoped>
/* 你的样式可以保留，或者根据需要调整 */
.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}

/* 可以添加一些基本的全局布局样式 */
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>