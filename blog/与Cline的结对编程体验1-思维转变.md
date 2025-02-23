
2025-02-16 22:25 

本周连着使用了大概15$的openrouter额度，在实现一个想法，写写side-project。这里记录一下自己使用Cline，Roo-Code的一些体会

> Roo-Code使用的是Anthropic最好的sonnet模型，上下文在200k（大大大！而且各方面能力都好，这也是Cline，Roo-Code这样项目令人惊艳的很大一部分原因）。  
> Roo-Code/Cline的system prompt，长度在 13k-15k token大小

一瞬间惊艳的感觉：当我给Cline一个链接，叫它给我构建一个MCP工具的时候。看着它显示开浏览器，然后访问对应的网址，然后拉取代码，构建代码。一步步的将perplexity的MCP工具构建好，虽然最后model name没有对（拉下来的代码model_name不能用），但是手动改后就能用了。这个时候给我一种惊艳的感觉，之前也有过想法，让LLM来构建自己需要的tool，但是没有继续往下想，或者说卡住了（仔细想一下为什么，差距在哪里？）


### 一些体会：
1. 当我的工程纯代码量在1500-2000之间的时候，上下文大小在30k左右的时候，绝大部分问题Roo-Code还是能够解决问题。
2. 当上下文到达40k-60k的时候，已经能够看到一些幻觉现象的，例如：有些潜在的依赖还是没有识别到
3. 在对话上下文稍微大一点后，虽然方案出的很快，但是如果无脑接收，还是有可能会出错的。因为每次返回的tool，可能会漏掉一些也有关系的文件（在上下文出于40-60k的观察到过）
4. 我们编码的时候，可能更需要注意一些范式（例如识别到耦合）。遇到过sonnet生成的代码，其实在松耦合方面还是有所欠缺的情况。解决办法，新开一个Task，然后@带上我们识别到的文件，叫它修改，就能解决了。
	1. 所以在大上下文的时候，我们如果还想要借助LLM的能力，这个时候review代码还是要关注一些编程范式。并且为了帮助它更好的解决问题，提高解决问题的概率，我们需要人工识别依赖，并且通过@关键文件的方式提供给LLM。
5. 瓶颈点，转移到了人脑。为什么这么说，因为虽然LLM能够缩小自然语言到外部世界的距离（言出法随，我说话，就改造了外部世界），但是并没有解决自然语言到脑子的距离（该不会，你还是不会，当前其实也加快了我们学习的速度）。也就是说，我们在前期的快速完成原型后，这个时候如果你看不懂这一大堆代码，那还是和以前开发一样的（前期爽是爽，但是看代码，debug问题，解决它解决不了的错误的时候，和以前是一样的），甚至因为快速膨胀的代码量，会给脑子带来更重的认知/思维负担。
6. 要有批评性思维/辩证思维，LLM输出的方案虽然快，能够把我的一个出发点和想法快速提供方案与实现。但是这其实也只是一个方向，如果我提的思路不行，那么它实现出来的可能也会差点意思。
	1. 解决办法：会写prompt，先切换Architect模式，先对话生成方案或者讨论方案，再切换Code模式，叫它给你编码
7. Cline的Memory Bank从某种方式，确实提供了助力。因为它将项目从三个高中低维度划分，并且落到文档中（越高，越偏向产品维度。越低，变动越频繁，越偏向落地实现的维度）。这样每次对话都会先去读对应层次的文档，但是也发现了一个问题，就是有些时候，新开的的Task不会去读Memory Bank。或者任务结束的时候，不会去更新Memory Bank。并且读取Memory Bank的时候是一个个的读的，每读一个都会发起一个LLM的请求带着以前一样的上下文继续问（如果模型API没有支持cache少扣钱的话，可能多来几次钱包就迅速爆炸了？）
8. **思维的转变**。有那么一瞬间，我突然有种我是老板的感觉，他是我手下的程序员。我以前从来没有这么想过，以前就是一个纯纯的技术思维，想的都是方案、实现、技术。但是这次在那一瞬间，我发现我思考的东西有点不一样了，虽然我也会去仔细review它的代码，但是我会多分一点心去考虑代码是否规范，是否复杂度上升了，后续扩展性这些问题。在它给的代码又快又好的时候，真的就是眼前一亮。在因为上下文逐渐变多，一个新的Task吃了0.5$，完成了一个我也能明显快速完成的任务的时候（会比它多花一点点时间），也会感觉有点心疼hhh……

从使用过程中体会到思维的些许改变，再到一些触动，让我感觉每个程序员可能都需要很好的去学会使用这些工具。但是我们不能依赖，否则瓶颈还是在脑子。

### 一些想法：
1. 可能很多人现在需要的是，align to llm，也就是和大模型对齐。你给它一个思路的时候，是怎么快速给出方案的，它是怎么想的（我们可能需要先思考这个问题，再去看它的答案体会），区别在哪里？否则离了大模型，你还是原来那样，原地踏步，甚至思考解决问题的能力降低了。
2. 在使用的过程，要"深度求索"，而不是"使用深度求索"，不能全信，要辩证批评的看待。
