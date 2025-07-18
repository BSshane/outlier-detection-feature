# 构建阶段 - 使用 Node.js 20
FROM docker.1ms.run/library/node:latest AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
# 添加 OpenSSL 兼容参数
RUN NODE_OPTIONS=--openssl-legacy-provider npm run build

# 生产阶段
FROM docker.1ms.run/library/nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]