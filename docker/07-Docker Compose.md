Docker Compose，它能够在 Docker节点上，以单引擎模式（ Single-Engine Mode）进行多容器应用的部署和管理
多数的现代应用通过多个更小的服务互相协同来组成一个完整可用的应用。 比如一个简单的示例应用可能由如下 4 个服务组成

- Web 前端
- 订单管理
- 品类管理
- 后台数据库

部署和管理繁多的服务是困难的。而这正是 Docker Compose 要解决的问题，**并不是通过脚本和各种冗长的 docker 命令来将应用组件组织起来， 而是通过一个声明式的配置文件描述整个应用，从而使用一条命令完成部署**
## 安装 Docker Compose 
直接使用`pip`安装，或者下载二进制文件
```shell
# pip 
pip install docker-compose -i https://pypi.tuna.tsinghua.edu.cn/simple

# download binary file
curl -L \
	https://github.com/docker/compose/releases/download/1.18.0/docker-compose-`\
	uname -s`-`uname -m` \
	-o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# test
docker-compose --version
# out
# docker-compose version 1.29.2, build unknown
```
## Compose 文件
Docker Compose 使用 YAML 文件来定义多服务的应用。 YAML 是 JSON 的一个子集，因此也可以使用 JSON
> YAML文件：**同一级的字段要对齐，冒号后面要带上空格**
> [https://zhuanlan.zhihu.com/p/340937902](https://zhuanlan.zhihu.com/p/340937902)

如下是一个简单的 Compose 文件的示例，它定义了一个包含两个服务（ web-fe 和 redis）的小型 Flask 应用。这是一个能够对访问者进行计数并将其保存到 Redis 的简单的 Web 服务。
```yaml
version: "3.5"
services:
  web-fe:
    build: .
    command: python app.py
    ports:
      - target: 5000
        published: 5000
    networks:
      - counter-net
    volumes:
      - type: volume
        source: counter-vol
        target: /code
  redis:
    image: "redis:alpine"
    networks:
      counter-net:

networks:
  counter-net:

volumes:
  counter-vol:
```
它包含 4 个一级 key： version、services、 networks、 volumes

1. `version` 是必须指定的，而且总是位于文件的第一行。它定义了 **Compose 文件格式**（主要是 API）**的版本**。建议使用最新版本, 并非定义 Docker Compose 或 Docker 引擎的版本号
2. `services` 用于定义不同的应用服务。上边的例子定义了两个服务：
- web-fe 的Web 前端服务
- redis 的内存数据库服务

Docker Compose 会将每个服务部署在各自的容器中

3. `networks` 用于指引 Docker 创建新的网络。默认情况下， Docker Compose 会创建 bridge 网络。这是一种单主机网络，只能够实现同一主机上容器的连接, 可以使用 `driver` 属性来指定不同的网络类型

下面的代码可以用来创建一个名为 over-net 的 `Overlay` 网络，允许独立的容器（ standalone
	container）连接（ attachable）到该网络上 
```yaml
networks:
  over-net:
  driver: overlay
  attachable: true
```

4. `volumes` 用于指引 Docker 来创建新的卷

在上面的示例文件中：
Compose 文件使用的是 v3.5 版本的格式，定义了两个服务，一个名为counter-net 的网络和一个名为 counter-vol 的卷
services 部分定义了两个二级 key： web-fe 和 redis
Docker Compose 会将每个服务部署为一个容器，并且会使用 key 作为容器名字的一部分。本例中定义了两个 key： web-fe 和 redis。因此 Docker Compose 会部署两个容器，一个容器的名字中会包含 web-fe，而另一个会包含redis 
在`web-fa`中

1. `build`： .指定 Docker 基于当前目录（ .）下 Dockerfile 中定义的指令来构建一个新镜
像。该镜像会被用于启动该服务的容器
2. `command`： python app.py 指定 Docker 在容器中执行名为 app.py 的 Python 脚本作为主程序，镜像中需要有对应的依赖, Dockfile如下

![image.png](https://51catgithubio.oss-cn-beijing.aliyuncs.com/1672723505024-6316f666-68e9-4803-b6fe-30caf6cf9b84.png)

3. `ports`：指定 Docker **将容器内（ **`**-target**`**）的 5000 端口映射到主机（published）的5000 端口**。这意味着发送到** Docker 主机 5000 端口的流量会被转发到容器的 5000 端口**。容器中的应用监听端口 5000
4. `networks`： 使得 Docker 可以将服务连接到指定的网络上。** 这个网络应该是已经存在的，或者是在 networks 一级 key 中定义的网络.**
5. `volume`指定 Docker 将 counter-vol 卷（ source:）挂载到容器内的/code（ target:）。 **counter-vol 卷应该是已存在的，或者是在文件下方的 volumes 一级key 中定义的**
> 综上， Docker Compose 会调用 Docker 来为 web-fe 服务部署一个独立的容器。该容器基于与 Compose 文件位于同一目录下的 Dockerfile 构建的镜像。基于该镜像启动的容器会运行app.py 作为其主程序，将 5000 端口暴露给宿主机，连接到 counter-net 网络上，并挂载一个卷到/code

在`redis`中：

1. `image`: redis:alpine 使得 Docker 可以基于 redis:alpine 镜像启动一个独立的名为 redis 的容器。这个镜像会被从 Docker Hub 上拉取下来
2.  `networks`：配置 redis 容器连接到 counter-net 网络

由于两个服务都连接到 counter-net 网络，因此它们可以通过名称解析到对方的地址
## 使用 Docker Compose 部署应用 
counter-app 的目录结构如下
```shell
tree ./counter-app-master/
./counter-app-master/
├── app.py
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```
启动compose应用的常用命令如下，它会构建所需的镜像，创建网络和卷，并启动容器, 自动寻找目录下的`docker-compose.yml` 或`dockercompose.yaml` 的 Compose 文件，其他文件则需要`-f`参数指定，使**用**`**-d**`**参数可以在后台启动应用，**Docker Compose 会将项目名称（ counter-app）和 Compose 文件中定义的资源名
称（ web-fe）连起来，作为新构建的镜像的名称
```shell
docker-compose up
```
使用以下命令启动示例的dock compose应用
```shell
docker-compose -f ./docker-compose.yml up  -d
```
查看一下：
```shell
docker image ls
# out
# REPOSITORY                  TAG          IMAGE ID       CREATED       SIZE
# counter-app-master_web-fe   latest       e1928444c34f   2 hours ago   84.6MB
# redis                       alpine       8ace02fae412   2 weeks ago   29.9MB

docker container ls
# out
# CONTAINER ID   IMAGE                       COMMAND                  CREATED       STATUS          PORTS                                       NAMES
# db44a4202599   counter-app-master_web-fe   "python app.py"          2 hours ago   Up 40 seconds   0.0.0.0:5000->5000/tcp, :::5000->5000/tcp   counter-app-master_web-fe_1
# 80713b5a3c33   redis:alpine                "docker-entrypoint.s…"   2 hours ago   Up 40 seconds   6379/tcp                                    counter-app-master_redis_1
```
查看网络和卷
```shell
docker network ls
```
![image.png](https://51catgithubio.oss-cn-beijing.aliyuncs.com/1672723847703-26d63fc4-fc83-4e74-baab-4ea0fea31bf8.png)
```shell
docker volume ls
```
![image.png](https://51catgithubio.oss-cn-beijing.aliyuncs.com/1672723880615-4e50a477-752b-44a3-92b5-08f7ba9ed4a4.png)
使用下面的命令结束docker-compose, 并删除容器
```shell
docker-compose down
```
使用下面的命令查看各个容器内运行的进程 
```shell
docker-compose top
```
使用下面的命令停止应用，但并不会删除资源
```shell
docker-compose stop
```
查看运行状态
```shell
docker-compose ps
```
删除compose应用
```shell
docker-compose rm
```
重启compose应用
```shell
docker-compose restart
```
