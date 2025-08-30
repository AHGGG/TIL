# 单词哥-英语理解力训练应用
> https://github.com/AHGGG/vibe-english-understanding

把单词哥的英语理解训练方法vibe-coding搞出来了

在线访问: [https://vibe-english-understanding.fun](https://vibe-english-understanding.fun)
- 训练步骤原文请在小红书搜索“单词哥”，或者访问本仓库的文件：[理解力提升操作步骤-单词哥](https://github.com/AHGGG/vibe-english-understanding/blob/master/%E7%90%86%E8%A7%A3%E5%8A%9B%E6%8F%90%E5%8D%87%E6%93%8D%E4%BD%9C%E6%AD%A5%E9%AA%A4-by%E5%8D%95%E8%AF%8D%E5%93%A5.md) 

# What i learned
1. 记得先用/init初始化整个工程的摘要信息。claude每次会带上摘要信息。不需要再去读一些文件了。

## 上下文工程
虽然cluade 200k，kimi 130+K，但是实际使用过程中，其实稍微多读几个文件，上下文就不够用了。

我的解决办法：
1. 最早的版本使用的css，一共7个step，每个step都有样式文件。将css重构为tailwind css，进行一次压缩。
2. 第二版，引入组件库，使用了更高级的抽象，也减小了上下文的依赖
3. 使用/compact命令来压缩。
4. 业务上进行解耦与原子的封装。在上下文触达边界的时候，如果它搞不定，最后就需要人工来进行任务的拆分、解耦。

## 使用vercel部署
虽然直接使用的vercel部署的国内无法直接访问。但只要注册了一个域名，转的cloudflare，然后绑定到vercel。试了下就能访问了。

# 一些体会
1. 虽然vibe coding吹得很厉害，但是只有真正去做，才能感觉到里面的问题。比如：稍微读几个文件了，上下文就不够用了。除了claude的工具调用错误会少点，kimi-v2，deepseek-v3.1的工具错误还是能经常看到的。
2. taste很重要。之前看到有人提过这个观点，往后走，在llm的辅助下代码可能大家都能写一点。这个时候taste就很重要了。比如：代码风格、架构设计、技术选型、业务理解、用户体验等。这些问题会更快的被遇到。
