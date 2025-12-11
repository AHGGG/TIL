1. 默认的plan模式规划文档和最后的review文档, 都会写入到另一个用户数据目录
	1. Task：TODO
	2. Implementation Plan：规划文档
	3. WalkThrough：这一次任务做了啥的文档
2. AI输出的文字中, 包含了文件、函数块的引用，点击可以跳到对应的文件并且选中对应的行或者块
3. 思考是英文，回答是中文
	1. 思考的结构是一个标题+内容这样的结构。兼顾了交互和效果。
4. 没有CLAUDE.md、AGENT.md这样文档，效果会差很多，需要手动维护到rule中让agent自己去读，这样就好了
5. 有一个专门的agent面板，可以多个任务入队列
6. browser的各种点击操作是通过一个sub agent来使用的（根据thinking的内容看出来的）
	1. 这种sub agent无法写文件
	2. browser的能力是通过安装一个插件来实现的，manus也是这样来实现的。看执行路径，应该是识屏，通过坐标进行操作。然后配合注入dom_id来获取dom的信息
	3. 
7. RunCommand可以由LLM添加超时设置，自己来决定哪些命令需要等多久
8. 读写文件采用的是按行读写
9. Gemini 3 Pro在运行命令的时候，经常写成Linux平台的命令
10. 在遇到问题进行Debug的时候，会自己写测试文件，测试完了删除
11. 有后台模式，可以
12. Claude Sonnet解决不了的问题，Gemini 3 Pro能解决。Gemini 3 Pro解决不了的问题Claude Opus 4.5切换能解决
13. 谷歌真是大善人，Gemini 3 Pro和Claude Opus 4.5无限额免费用
