<template>
  <el-dialog v-model="dialogVisible" width="512px" @close="closeDialog" destroy-on-close>
        <slider-verify @onSuccess="success" @onError="error" :width="SLIDER_WIDTH" :height="SLIDER_HEIGHT"
      :verifyPass="actualVerifyPass" :img="verifyImage">
    </slider-verify>
  </el-dialog>
</template>

<script setup>
import { ElMessage } from "element-plus";
import { ref } from 'vue';

const emit = defineEmits(['onSuccess', 'onError', 'onClose']);

const dialogVisible = ref(false);
// 滑块图片
const verifyImage = ref('');

// **新增：控制 slider-verify 组件的 verifyPass 状态**
const actualVerifyPass = ref(false); // 初始为 false，表示不进行验证

// 配置常量
const IMAGE_BASE_PATH = '/images/verifyImages/';
const IMAGE_EXTENSION = '.jpeg';
const IMAGE_COUNT = 4;
const SLIDER_WIDTH = 480; // 320 * 1.5
const SLIDER_HEIGHT = 270; // 180 * 1.5

/**
 * 成功回调
 */
const success = (verifyResult) => {
  dialogVisible.value = false;
  actualVerifyPass.value = false; // 验证成功后重置
  emit('onSuccess', verifyResult);
};

/**
 * 失败回调
 * @param verifyResult
 */
const error = (verifyResult) => {
  ElMessage.error(verifyResult?.message || '验证失败');
  emit('onError', verifyResult);
  // actualVerifyPass.value = false; // 失败后不重置，让用户可以重新尝试，如果需要重置，请取消注释
};

/**
 * 随机获取预备的验证图片
 * @returns {string}
 */
const generateVerifyImg = () => {
  const index = Math.floor(Math.random() * IMAGE_COUNT) + 1;
  return `${IMAGE_BASE_PATH}bg${index}${IMAGE_EXTENSION}`;
};

/**
 * 打开弹窗
 */
const openDialog = () => {
  dialogVisible.value = true;
  actualVerifyPass.value = true; // **弹窗打开时才激活验证**
  verifyImage.value = generateVerifyImg();
};

/**
 * 关闭弹窗
 */
const closeDialog = () => {
  dialogVisible.value = false;
  actualVerifyPass.value = false; // **弹窗关闭时取消激活验证**
  emit('onClose');
};

// 暴露方法供外部调用
defineExpose({
  openDialog,
  closeDialog
});
</script>

<style scoped>
/* 添加一些简单的样式 */
.el-dialog__body {
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>