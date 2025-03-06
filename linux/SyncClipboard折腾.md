问题: 
1. 每次手机拷贝了东西, 发到电脑上(Ubuntu)不方便, 有些敏感信息不想通过第三方的服务器
2. 之前每次都是使用的LocalSend, 但是每次都需要在一个局域网下, 不方便
3. 看到同事iphone直接拷贝了, 连的公司的wifi, 也能直接在电脑上粘贴(发现苹果电脑粘贴的时候鼠标转圈圈了0.5s-1s的样子, 估计是从服务器在拉取), 所以也准备自己搭建一个

### 安装nginxproxymanager
```
services:
  app:
    image: 'jc21/nginx-proxy-manager:latest'
    restart: unless-stopped
    ports:
      - '80:80'
      - '81:81'
      - '443:443'
    volumes:
      - ./data:/data
      - ./letsencrypt:/etc/letsencrypt
      - ./mysql:/var/lib/mysql
```
``

问题: 
1. docker拉取不到, 解决办法一: [配置镜像](https://www.coderjia.cn/archives/dba3f94c-a021-468a-8ac6-e840f85867ea), 解决办法二: 配置代理, 让docker pull走代理
2. 登录的时候报502 bad gateway, 解决办法: [bad_gateway_on_nginxproxymanager_running_in_docker](https://www.reddit.com/r/nginxproxymanager/comments/12ilet7/bad_gateway_on_nginxproxymanager_running_in_docker/)

### 安装syncClipboard
```
version: '3'
services:
  syncclipboard-server:
    image: jericx/syncclipboard-server:latest
    container_name: syncclipboard-server
    restart: unless-stopped
    ports:
      - "5033:5033" # Update this if you have changed the port in appsettings.json
    environment:
      - SYNCCLIPBOARD_USERNAME=xxxxUSERNAME
      - SYNCCLIPBOARD_PASSWORD=xxxxPASSWORD
```

### 前置工作
购买域名
配置dns解析
然后检查自己的dns解析是否指向了自己的服务器, 工具: https://tool.chinaz.com/dns/
### 申请证书
nginx proxy manager里新增一个proxy host, 配置如下:
1. 添加自己的域名, http, Forward Hostname / IP(通过ifconfig看到docker所在的内网地址), 端口填跑起来的syncClipboard端口5033, 勾选Block common Exploits
2. 然后SSL tabl, 选择申请新的证书, 勾选Force SSL, HTTP/2 support, HSTS support
3. 点击确定, 如果失败了, 再确定一下, 就成功了

> 之前也试过自己配置nginx, 然后申请证书, 或者使用acme官网提供的通过dns的方式生成: 参考cloudflare: https://github.com/acmesh-official/acme.sh/wiki/dnsapi#dns_cf


### 配置SyncClipboard
windows, [下载软件](https://github.com/Jeric-X/SyncClipboard?tab=readme-ov-file#windows), 然后启动的时候安装几个需要的依赖, 然后还有字体.
配置: 
1. 开启剪贴板同步, 修改地址为我们申请的域名, 例如https://sync.xxx.xx
2. 用户名填前面你修改过的xxxxUSERNAME, 密码同理
3. 然后在服务状态中检查一下状态信息, 是否是running就行了
