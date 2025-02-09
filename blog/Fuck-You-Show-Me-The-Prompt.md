## Fuck-You-Show-Me-The-Prompt（绕过项目和框架，直接拦截LLM接口）
之前看到了这篇文章[Fuck You, Show Me The Prompt](https://hamel.dev/blog/posts/prompt/)，被里面配的图笑到了。
其实就是不管我们吹一个产品有多牛，但是其实本质上就是调了一个大模型的API，然后再就是工程手段，让调模型的请求体动态的构造出来。

### 问题
但是各种框架都封装了太多层，调用一个大模型的API，要写一大段的代码。美其名曰：封装、优雅、高扩展性。其实有的时候并不需要这些复杂的框架，这也是之前看到有些初创团队放弃langchain的原因。就是比起解决的问题，这些框架引入了更多的复杂度，或者说引入了更多的认知负担（参考：[Cognitive Load is what matters](https://github.com/zakirullin/cognitive-load)）

之前尝试看blot.diy的代码，想看看它是怎么实现一句话生成页面的，和之前的[react-agent](https://github.com/eylonmiz/react-agent), [openv0](https://github.com/raidendotai/openv0)有什么不同的。
但是clone到本地，就直接被代码的复杂度掩盖住了双眼。首先就是熟悉代码结构，装一下依赖，这样代码可以索引到然后支持引用与跳转。花了点时间，找到了prompt，然后看看prompt是怎么定义的，然后看看注入哪些属性，这些属性又是从哪里来的。
这个时候我就想吐槽一句，我只想看一下你请求大模型接口的时候请求体是什么，请求体组装出来是什么样子，多轮交互的时候请求体又是什么样，返回的又是什么（这时候表现是什么样的）。这样我就能对快速对这个项目的实现有个大概的了解，这个时候如果还想继续深入细节（比如动态构造的时候有哪些考虑因素），就可以慢慢去看看代码。但是，我们很多时候都是这样被代码的逻辑困住了脚步。

### 解决方案
而[Fuck You, Show Me The Prompt](https://hamel.dev/blog/posts/prompt/)这篇文章就提到了一个解决方案，我只需要拦截LLM的API请求就行了，这样就能看到调LLM接口的时候请求体，prompt长什么样。

拦截LLM有很多种方式，可以是：path源代码、代理。代理的意思就是，我们在本地构建一个http/https的代理，这样请求会先经过我们构建的代理，这样就能看到请求的各种信息。
我理解其实就是实现中间人攻击，安装好证书后，我们就可以解密https的数据，看到请求体信息。
> 手机使用Reqable，同样也是跑一个代理，安装证书，然后抓包，拿到cookie之后可以，，看我抽卡记录。。。hhh之前这么搞过

### 实践
最近在用Vscode的Roo-Code插件（从Cline fork的一个仓库），我想知道它在给我生成代码的时候，用了哪些信息，prompt长什么样。

#### 前置条件
我本地装有代理，平常有的时候用系统代理，有时候开启TUN模式。
有windows和Ubuntu两种环境。

#### 安装mitmproxy
跟着官网的doc，安装好了

打开mimweb，就会弹出ui界面。这个时候`mitmproxy`就运行在了本地的8080端口。然后安装`mitmproxy`的CA证书，`mitmproxy`给各个平台都提供了安装指引。

这个时候需要我们设置一下系统代理，linux可以通过ui界面里设置。windows可以搜索代理服务器设置，设置我们的系统代理。
> 需要注意的时候，有些程序不会走系统代理，例如cmd。下面也遇到了这个问题。

#### 修改vscode的proxy配置
进行setting，
1. 修改vscode的proxy为`mitmproxy`监听的`http://127.0.0.1:8080`（后面尝试改成`http://127.0.0.1:7897`也可以，就是不能为空...），
2. 修改Proxy support为On
3. 修改Proxy Strict SSL改为不启用

通过wireshark抓包，看到Roo-Code配置了openrouter提供的接口后，请求也是请求的`https://openrouter.ai/api`，但是无法解密的，因为vscode不是浏览器环境，没有使用我们的pre-master secret，wireshark自然也是看不到了。
> 就算我们配置了pre-master secret，wireshark也只能看到Chrome浏览器发起的https请求，因为vscode里的这个插件，不是浏览器环境！在github web编辑页面也无法使用Roo-Code，因为没有File API。

我猜测Roo-Code使用了浏览器的fetch，或者node-fetch这样的库。这样的库如果不自动判断是否走系统代理，那么我们通过`mitmproxy`设置的代理，其实就跟没有一样。

#### 重定向浏览
wireshark抓包，找到Roo-Code就是直接发往**openrouter.ai**的请求。所以我们现在的目标就是将发往**openrouter.ai**的请求，转发到本地的`127.0.0.1:8080`的代理上去。
1. 安装`proxifier`这个软件
2. 配置**proxy server**为`127.0.0.1:8080`
3. 配置**proxification rules**，将**openrouter.ai**的请求，也转发到`127.0.0.1:8080`
4. 重启一下vscode

#### 测试
打开Roo-Code插件，测试输入，发起请求，这个时候`proxifier`会看到对应的rule的log，**openrouter.ai**的请求被转发到了8080端口。这个时候打开`mitmproxy`的web页面，就可以看到发往**openrouter.ai**的请求了，也是解密好的。
点开一条记录，可以正常看到request，response。然后看了下Roo-Code的system prompt，使用的Anthropic的token计算器，12000+的token（哭...）

#### 其他
1. 这个时候打开本地的TUN代理，也是都正常的。
2. `mitmproxy`还支持透密代理，但是windows不支持。


后面如果想要看一个项目的请求体的时候，就可以这样，`mitmproxy`启用一个代理，如果这个项目用的请求库能够自动判断是否走系统代理，这个时候`mitmproxy`的web界面就能直接看到。但是，如果这个项目程序是直接发往目标地址的，就可以使用`proxifier`，将这个请求转发到`mitmproxy`搭建的代理，这样也能看到了

### reference
https://hamel.dev/blog/posts/prompt/
[Cognitive Load is what matters](https://github.com/zakirullin/cognitive-load)
[react-agent](https://github.com/eylonmiz/react-agent)
[openv0](https://github.com/raidendotai/openv0)
