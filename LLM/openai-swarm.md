### Swarm
openai新开源的multi-agent调用框架. 提供轻量, 可扩展, 高度自定义支持的多agent协作模式.

### 使用场景
处理存在大量独立功能和指令的情况，这些功能和指令很难放到单个prompt

### 使用示例
```python
# agent定义
sales_agent = Agent(name="Sales Agent")

# tool定义, 返回结果+额外的context
def talk_to_sales():
   print("Hello, World!")
   return Result(
       value="Done",
       agent=sales_agent,
       context_variables={"department": "sales"}
   )

# 带functions的agent定义
agent = Agent(functions=[talk_to_sales])

# 调用agent, 会调用talk_to_sales这个函数, 然后继续调用sales_agent再返回
response = client.run(
   agent=agent,
   messages=[{"role": "user", "content": "Transfer me to sales"}],
   context_variables={"user_name": "John"}
)
print(response.agent.name)
print(response.context_variables)
```


### 两个核心概念:
#### 1. Routine(Agent): 
可以理解成包含一些步骤+一些工具调用的处理流程. routine其实也可以理解成agent/sssistant. 和那张非常经典的agent组成部分图一样
1. Routine的instruction中包含: 自然语言描述的分支处理逻辑(任务规划). 例如如果xxx, 那么调用xx处理. 如果xxx了, 那么做xxx, 例如下面的这个system_message
```
system_message = (
    "You are a customer support agent for ACME Inc."
    "Always answer in a sentence or less."
    "Follow the following routine with the user:"
    "1. First, ask probing questions and understand the user's problem deeper.\n"
    " - unless the user has already provided a reason.\n"
    "2. Propose a fix (make one up).\n"
    "3. ONLY if not satesfied, offer a refund.\n"
    "4. If accepted, search for the ID and then execute refund."
    ""
)
```

另一个[examples/personal_shopper/main.py#L115](https://github.com/openai/swarm/blob/main/examples/personal_shopper/main.py#L115):
```
	You are to triage a users request, and call a tool to transfer to the right intent.(任务顶层说明)
    Once you are ready to transfer to the right intent, call the tool to transfer to the right intent.(意图分发顶层说明)
    You dont need to know specifics, just the topic of the request.(约束)
    If the user request is about making an order or purchasing an item, transfer to the Sales Agent.(意图分支)
    If the user request is about getting a refund on an item or returning a product, transfer to the Refunds Agent.(意图分支)
    When you need more information to triage the request to an agent, ask a direct question without explaining why you're asking it.(追问功能添加)
    Do not share your thought process with the user! Do not make unreasonable assumptions on behalf of user(约束)
```
- 任务说明(意图分发说明)+意图分支+追问+约束, 有这个instruction的agent会作为最外层的agent最先开始执行

整体调用流程:
```python
messages = []
while True:
	1. 调用llm接口, 获取返回
	2. 判断如果没有tool调用 ==> break
	3. 处理tool_call, 然后将返回的消息放到messages最后
```

#### 2. handoffs:
因为单个Routine处理能力有限, 所以如果把多个任务交给一个Routine来处理, 它会显得力不从心. 
> tools过多, instruction完成的事情太多, 力不从心.

所以这里引入了handoffs, 实现动态的切换instructions+tools. 一个Routine(Agent)将对话转交给另一个Routine(Agent)
> 隐含: 怎么划分不同的Routine(Agent, 包含合适的tool), 怎么切分足够合适大小的Routine, 怎么对业务领域的问题进行子问题切分? 如果切分不了, 要怎么写?

现在整体调用流程变成了:
```python
messages = []
active_agent = init_agent
while True:
	1. 调用active_agent(包含不同的instruction+tools)
	2. 判断如果没有tool调用 ==> break
	3. 处理tool_call, 然后将返回的消息放到messages最后
	4. 如果返回的tool_call返回的是agent, 那么更新上下文的active_agent, 继续回到步骤1进行执行
```

### 问
1. 怎么从大模型返回的内容中判断是要执行一个函数, 还是要执行一个agent呢?  
其实对于大模型来说, 都是一个个的tool. 在大模型判断需要调用function的时候, 都会返回function call(一个json, 描述了要调用的函数的名字+函数的参数). 从中可以解析出function的名字, 拿到tool_map然后调用function, 然后判断函数返回的类型其实就行了. 返回的是Agent类型的, 那么代表下面应该交接给另一个agent进行处理

2. messages/chat_history/对话上下文是怎么在多个agent之间流转的?  
- 计算刚开始history的长度
- 不管当前agent返回了function_call或者纯文本, 那么都放到history最后
- function_call返回的消息, 也继续放到到history最后
- while循环结束, 根据第一步计算的初始history长度, 将新增的n条messages返回
- history数组中最后一个content就是整个调用链路的最终返回
- 所以每一轮(while开启新的循环), 如果调用一个新的agent, 传进行的history都会包含与前面n个agent交互的所有上下文

3. tool如果想要修改上下文, 或者返回其他的内容怎么办?  
一个函数返回的Result结构体中, 可以返回context_variables, tool返回的context_variables会被更新到一个顶部声明好的context_variables字典中. 而顶部声明好的context_variables会被包装到本次执行结果中一起返回.

4. 这个框架有什么新东西, 有什么不同?  
将tool_call返回的内容进行扩展, 不仅仅限制于返回字符串, 返回各种外界交互的结果. 还可以返回一个agent, 这样配置将tool的名字改成`transfer_to_xxx`, 配置将instructions中添加说明`任务说明(意图分发说明)+意图分支+追问+约束`. 这样就实现了在满足xx条件的时候, 调用`transfer_to_xx`返回一个agent, 然后继续调用这个agent, 直到返回的不是function_call

5. 测试类怎么写的?  
- 用好run_demo_loop

6. 如果tool想要接受context_var要怎么办?  
tool可以声明context_var的参数, 然后代码中会通过`func.__code__.co_varnames`来判断是否有声明需要context_var的参数. 如果有的话, 把从顶部一路透传的context_var赋值给对应的参数, 这样调用tool就会正确拿到上下文参数

7. 整个项目的结构?  
整个项目非常简单, 就一个核心的core类, 其他的都是一些utils
### reference: 
[Orchestrating Agents: Routines and Handoffs, 其实这篇文章里已经说的很清楚了](https://cookbook.openai.com/examples/orchestrating_agents)
https://github.com/openai/swarm?tab=readme-ov-file
