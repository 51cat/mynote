# Docker

简洁版

```shell
# 启动
systemctl start docker

# 以ubuntu:latest 为例子
# 拉取镜像
docker image pull ubuntu:latest

# 搜索镜像
docker search ubuntu

# 查看镜像
docker image ls
docker image ls --filter reference="ubuntu*" # 使用通配符

# 查看容器
docker container ls

# 启动容器
docker container run --name percy2 -it ubuntu:latest /bin/bash

# 进入容器
docker container exec -it percy2 /bin/bash

# 退出容器
Ctrl + P + Q (快捷键) 
exit

# 停止容器
docker container stop  percy2

# 重启容器
docker container stop percy2
docker container start percy2

# 删除容器
docker container rm percy2

# 删除镜像
docker image rm  ubuntu:latest
docker image rm 6852022de69d

# 删库跑路用↓↓↓↓ (不是
# 停止全部容器
docker container stop $(docker container ls -qa)
# 删除全部容器
docker container rm $(docker container ls -qa) -f
# 删除全部镜像(用容器依赖的不会删除)
docker image rm $(docker image ls -q) -f
```

## 修改镜像

### 启动容器

```
docker container run --name percy2 -it ubuntu:latest /bin/bash`
```

### 容器内一些操作,

`...`

### 退出容器

```shell
exit
```

### commit

```shell
docker commit -m "do something " -a "Author Name" <container id> <new image name>:<tag>
```

例如

```shell
docker commit -m "add python package colorlog " -a "liuzihao" 82731c189c3b  gim3e:v4
```

![image-20231227144459075](https://51catgithubio.oss-cn-beijing.aliyuncs.com/image-20231227144459075.png)

## 容器内外复制文件

```shell
docker cp /path/filename 容器id或名称:/path/filename
docker cp 容器id或名称:/path/filename /path/filename
```

## 导出/导入镜像

## export / import

### 导出

```shell
# exorpt
docker export IMAGE ID > test.tar
# save 
docker save IMAGE ID > test.tar
# 多个save
docker save -o test.tar postgres:9.6 mongo:3.4
```

### 导入

```shell
# import
docker import <image name> < test.tar
# load 
docker load < test.tar
```

## 区别

不可混用！

1. **文件大小不同**

**export** 导出的镜像文件体积小于 **save** 保存的镜像

2. **是否可以对镜像重命名**

- **docker import** 可以为镜像指定新名称
- **docker load** 不能对载入的镜像重命名

3. 是否可以同时将多个镜像打包到一个文件中

- **docker export** 不支持
- **docker save** 支持

5. **是否包含镜像历史**

- **export** 导出（**import** 导入）是根据容器拿到的镜像，再导入时会丢失镜像所有的历史记录和元数据信息（即仅保存容器当时的快照状态），所以无法进行回滚操作。
- 而 **save** 保存（**load** 加载）的镜像，没有丢失镜像的历史，可以回滚到之前的层（**layer**）。

6. **应用场景不同**

- **docker export 的应用场景**：主要用来制作基础镜像，比如我们从一个 **ubuntu** 镜像启动一个容器，然后安装一些软件和进行一些设置后，使用 **docker export** 保存为一个基础镜像。然后，把这个镜像分发给其他人使用，比如作为基础的开发环境。
- **docker save 的应用场景**：如果我们的应用是使用 **docker-compose.yml** 编排的多个镜像组合，但我们要部署的客户服务器并不能连外网。这时就可以使用 **docker save** 将用到的镜像打个包，然后拷贝到客户服务器上使用 **docker load** 载入。
