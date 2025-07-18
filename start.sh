# 检查并停止删除后端容器
if docker inspect backend-container > /dev/null 2>&1; then
    # 仅当容器正在运行时才停止
    if [ "$(docker inspect -f '{{.State.Running}}' backend-container 2>/dev/null)" = "true" ]; then
        docker stop backend-container
    fi
    docker rm backend-container
fi

# 检查并停止删除前端容器
if docker inspect frontend-container > /dev/null 2>&1; then
    # 仅当容器正在运行时才停止
    if [ "$(docker inspect -f '{{.State.Running}}' frontend-container 2>/dev/null)" = "true" ]; then
        docker stop frontend-container
    fi
    docker rm frontend-container
fi

# 检查并删除后端镜像
if docker inspect backend-image > /dev/null 2>&1; then
    docker rmi backend-image
fi

# 检查并删除前端镜像
if docker inspect frontend-image > /dev/null 2>&1; then
    docker rmi frontend-image
fi
# 构建后端镜像
docker build -t backend-image ./ServerPage
# 构建前端镜像
docker build -t frontend-image ./ServerPage/frontend
# 运行后端容器
docker run -d -p 9080:9080 --restart=always --name backend-container backend-image
# 运行前端容器
docker run -d -p 9094:80 --restart=always --name frontend-container frontend-image