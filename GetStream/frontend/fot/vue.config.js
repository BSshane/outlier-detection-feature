module.exports = {
  lintOnSave: false,
  devServer: {
    port: 8080, 
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        pathRewrite: { '^/api': '' }
      }
    }
  }
}