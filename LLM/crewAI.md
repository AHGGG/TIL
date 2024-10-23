## 文档学习
### memory
分为短期记忆/长期记忆/Entity Memory(先理解为: 从上下文中提取出的实体)/Contextual Memory

Contextual Memory: Maintains the context of interactions by combining ShortTermMemory, LongTermMemory, and EntityMemory, aiding in the coherence and relevance of agent responses over a sequence of tasks or a conversation.

Experience Accumulation: Long-term memory allows agents to accumulate experiences, learning from past actions to improve future decision-making and problem-solving.
> 有点感受, 需要agent有长期的记忆

Entity Understanding: By maintaining entity memory, agents can recognize and remember key entities, enhancing their ability to process and interact with complex information.
> 对于一些实体的总结, 其实就是对于领域实体进行建模, 描述!


## 代码学习
### Crew使用
```python
@CrewBase
class GameBuilderCrew:
    """GameBuilder crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def senior_engineer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['senior_engineer_agent'],
            allow_delegation=False,
            verbose=True
        )
    
    @agent
    def qa_engineer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['qa_engineer_agent'],
            allow_delegation=False,
            verbose=True
        )
    
    @agent
    def chief_qa_engineer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['chief_qa_engineer_agent'],
            allow_delegation=True,
            verbose=True
        )
    

    @task
    def code_task(self) -> Task:
        return Task(
            config=self.tasks_config['code_task'],
            agent=self.senior_engineer_agent()
        )

    @task
    def review_task(self) -> Task:
        return Task(
            config=self.tasks_config['review_task'],
            agent=self.qa_engineer_agent(),
            #### output_json=ResearchRoleRequirements
        )

    @task
    def evaluate_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluate_task'],
            agent=self.chief_qa_engineer_agent()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the GameBuilderCrew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True, 
        )

```

运行:
```python
game= GameBuilderCrew().crew().kickoff(inputs=inputs)
```

### CrewBase 类装饰器
```python
`def CrewBase(cls: T) -> T:
	class WrappedClass(cls):
		# 一些配置的属性定义...
		xxx
		xxx

		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			# 读取所有的config
			# 处理config中的agent variables
			# 处理config中的task variables

	# cast确保返回的类型符合原始类的类型，以便在类型检查时不会出错。
	return cast(T, WrappedClass)`
```

### utils
#### memoize
就是踹吃个装饰器, 将func的args, kwargs计算一个key. 如果key不在cache中, 那么调用这个函数, 否则直接返回.
memoize这个函数返回的是装饰后的函数, 对于原来的函数其实就是多个一个拦截处理, 其他都被原样拷贝了
```python
def memoize(func):
    cache = {}

    def memoized_func(*args, **kwargs):
        key = (args, tuple(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

	# 将原始func里的所有属性拷贝到新的函数中
    memoized_func.__dict__.update(func.__dict__)
    return memoized_func
```

### agent task crew装饰器
[source code](https://github.com/crewAIInc/crewAI/blob/main/src/crewai/project/annotations.py)
#### agent
```python
def agent(func):
    func.is_agent = True
    func = memoize(func)
    return func
```
这里手动调用了memoize函数, 装饰了一个原本的func函数. 
所以agent装饰器干的事情就是: 
1. 给被装饰的func添加一个`is_agent`为`True`的标识
2. 装饰了一个func, 添加上了cache功能. 如果参数是一样的, 命中了缓存, 会直接返回.

#### task
```python
def task(func):
    func.is_task = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not result.name:
            result.name = func.__name__
        return result

    return memoize(wrapper)
```
设置func的`is_task`属性为`True`
这里技术上有多层装饰的意思, 第一层装饰: 调用func后返回的结果, 如果没有name属性, 那么会将func调用结果中的name属性赋值为`func.__name__`(被装饰的func的名字)
第二层装饰: 给wrapper函数添加了缓存能力

#### crew
crew用在了CrewBase装饰的类里面的函数上面. 
crew是一个装饰函数的装饰器
```python
def crew(func) -> Callable[..., Crew]:
    def wrapper(self, *args, **kwargs) -> Crew:
		# 一些初始化操作
		return func(self, *args, **kwargs)

	return wrapper
```


wrapper内部:
1. 通过`self.__class__.__dict__.items()`拿到crew所在类里所有的方法和属性
- `self.__class__`: 拿到crew装饰器所在的Class定义
- `.__dict__`: 拿到crew装饰器所在class内部所有的属性和方法(包括实例属性和类属性)
2. 根据标识`is_agent`和`is_task`找到所有的agents, tasks
3. 实例化所有的agent和所有的task, 放到一个数组中
4. 将实例化后的所有放到self.agents和self.tasks中

#### 其他装饰器
剩余的其他装饰器, 看起来就是用memoize包装了一下, 设置了一下标志位.

### Crew类kickoff运行逻辑
1. 输入处理
2. 将crew的引用保存到每个内部的agent中
	1. 遍历每个agent的时候, 初始化几个属性: 
	2. `function_calling_llm`: LLM的实例, 在`@model_validator(mode="after")`内部进行初始化
	3. `allow_code_execution`: 如果允许的话, 会给agent的tool里注入一个执行代码的tool
	4. `step_callback`: 所有agent每个step执行后, 运行的callback. (补充另一个: `task_callback`: 所有agent每个task执行后, 运行的callback)
	5. 给每个agent初始化executor. 其实就是将这个agent相关的所有信息, 都封装到一个结构化的数据结构中(CrewAgentExecutor)
```python
self.agent_executor = CrewAgentExecutor(
	llm=self.llm,
	task=task,
	agent=self,
	crew=self.crew,
	tools=parsed_tools,
	prompt=prompt,
	original_tools=tools,
	stop_words=stop_words,
	max_iter=self.max_iter,
	tools_handler=self.tools_handler,
	tools_names=self.__tools_names(parsed_tools),
	tools_description=self._render_text_description_and_args(parsed_tools),
	step_callback=self.step_callback,
	function_calling_llm=self.function_calling_llm,
	respect_context_window=self.respect_context_window,
	request_within_rpm_limit=(
		self._rpm_controller.check_or_wait if self._rpm_controller else None
	),
	callbacks=[TokenCalcHandler(self._token_process)],
)
```
> `CrewAgentExecutor`, 核心的是`invoke`方法(内部调用了`_invoke_loop`), 负责最终调用llm

#### planning
3. crew如果设置了planning的参数, 那么会首先进行planning
```python
def _handle_crew_planning(self):
	"""Handles the Crew planning."""
	# 这一步完成planning
	result = CrewPlanner(
		tasks=self.tasks, planning_agent_llm=self.planning_llm
	)._handle_crew_planning()

	# 将planning的结果放到task的description中
	for task, step_plan in zip(self.tasks, result.list_of_plans_per_task):
		task.description += step_plan.plan
```
3.1 `_handle_crew_planning`函数处理逻辑:
- 初始化planning `Agent`
	- 就是创建一个`Agent`的实例
```python
Agent(
	role="Task Execution Planner",
	goal=(
		"Your goal is to create an extremely detailed, step-by-step plan based on the tasks and tools "
		"available to each agent so that they can perform the tasks in an exemplary manner"
	),
	backstory="Planner agent for crew planning",
	llm=self.planning_agent_llm,
)
```
- 构建tasks_summary
```python
def _create_tasks_summary(self) -> str:
	"""Creates a summary of all tasks."""
	tasks_summary = []
	for idx, task in enumerate(self.tasks):
		tasks_summary.append(
			f"""
			Task Number {idx + 1} - {task.description}
			"task_description": {task.description}
			"task_expected_output": {task.expected_output}
			"agent": {task.agent.role if task.agent else "None"}
			"agent_goal": {task.agent.goal if task.agent else "None"}
			"task_tools": {task.tools}
			"agent_tools": {task.agent.tools if task.agent else "None"}
			"""
		)
	return " ".join(tasks_summary)
```
- 构建一个planning `Task`
```python
Task(
	description=(
		f"Based on these tasks summary: {tasks_summary} \n Create the most descriptive plan based on the tasks "
		"descriptions, tools available, and agents' goals for them to execute their goals with perfection."
	),
	expected_output="Step by step plan on how the agents can execute their tasks using the available tools with mastery",
	agent=planning_agent,
	output_pydantic=PlannerTaskPydanticOutput,
)
```
> 注意这里output_pydantic指定了输出的结果, PlannerTaskPydanticOutput: 
```python
class PlanPerTask(BaseModel):
    task: str = Field(..., description="The task for which the plan is created")
    plan: str = Field(
        ...,
        description="The step by step plan on how the agents can execute their tasks using the available tools with mastery",
    )


class PlannerTaskPydanticOutput(BaseModel):
    list_of_plans_per_task: List[PlanPerTask] = Field(
        ...,
        description="Step by step plan on how the agents can execute their tasks using the available tools with mastery",
    )
```

##### Task的execute_sync方法
调用planning Task的`execute_sync`方法(委托给agent(Agent类)的`execute_task`方法). 然后各种初始化prompt/memory/tool/agent_executor。最后委托给agent_executor的invoke方法。
	- invoke方法内部调用了`_invoke_loop`方法，拿到`formatted_answer`
`_invoke_loop`方法内部是一个`while`, 如果formatted_answer不是`AgentFinish`类的实例, 那么就不结束!
- 内部先委托给self.llm, 调用大模型, 拿到返回值
- 然后正则ReAct的方式解析action和action input, 返回AgentAction/AgentFinish类, 赋值给`formatted_answer`
	- 兼容了多种异常的情况:
		- 同时有Action+Answer ==> 异常
		- 没有Action, 有Answer ==> AgentFinish
		- 没有Action返回 ==> 异常
		- 最后就是未知情况 ==> 异常
- 从大模型返回的纯文本中, 解析tool的调用(`self._use_tool(formatted_answer)`这里面, 做了非常多的脏活累活, 大概过了一下. 没有langchain的代码清晰). tool调用里面兼容了多种情况. 除了正常的调用, 还有:
	- tool名字不对
	- tool的名字只要85%匹配, 那么就算选择了这个tool ==> `SequenceMatcher(None, tool.name.lower().strip(), tool_name.lower().strip()).ratio() > 0.85`
	- tool参数校验
	- 调用ast模块, 将tool_input(字符串)通过`ask.literal_eval`来进行执行. 内部还有其他处理
	- 兼容了使用的是function_calling的情况
	- 最后都封装到一个对象`ToolCalling`里返回
- while最后判断, 如果没有force_return, 那么把这一次调用的结果, 放到messages最后, 继续while循环
	- 相当于tool调用后的结果(有Observation: `<tool调用的结果>`)放到messages最后作为assistant类型的消息(ReAct), 继续调用. 直到返回AgentFinish, 注意AgentFinish的output是最后正则解析出来的Final Answer, 一路返回, 最后包装成TaskOutput对象返回. 
		- 如果是tool调用, 支持通过`result_as_answer`直接返回结果
		- 如果返回的结果需要是结构化的, 也会尝试将返回的output解析为对象. 失败还是继续返回原始的output
- 调用拿到planning的结果. 回到最外层, 刚刚的output作为`planner_task`的返回, 如果不是pydantic对象, 那么会直接抛出ValueError说planning失败.
	- 如果返回的正常, 那么我们就拿到了规划后的信息: 
```python
[
 {
	 "task": "task的名字??",
	 "plan": "一步步的描述的执行计划, 关于agent要怎么借助现有的tool, 来执行他们的任务"
 }
]
```
> TODO: 还要再看, 还是有点迷


### 三种不同的Memory的处理逻辑
#### ContextualMemory

#### CrewAgentExecutorMixin
#####  `_create_short_term_memory`
逻辑很简单：
1. 看crew的`_short_term_memory`是否存在，如果存在的话，应该是一个ShortTermMemory类
2. 如果存在(被初始化了)，那么直接委托调用这个类的save方法。保存如下信息：==output.log==(没懂是什么，`_create_short_term_memory`这个项目中没有被调用到)，==metadata==（包括：self.task.description），==self.crew_agent.role==

##### `_create_long_term_memory`
- 调用`TaskEvaluator`的evaluate方法. 内部调用了大模型, 经过预制好的一段prompt
```python
evaluation_query = (
	f"Assess the quality of the task completed based on the description, expected output, and actual results.\n\n"
	f"Task Description:\n{task.description}\n\n"
	f"Expected Output:\n{task.expected_output}\n\n"
	f"Actual Output:\n{output}\n\n"
	"Please provide:\n"
	"- Bullet points suggestions to improve future similar tasks\n"
	"- A score from 0 to 10 evaluating on completion, quality, and overall performance"
	"- Entities extracted from the task output, if any, their type, description, and relationships"
)
```

经过前面的prompt, 产出entity定义`suggestions`, `quality`, `entities`。也就是提取出实体。
```python
class Entity(BaseModel):
    name: str = Field(description="The name of the entity.")
    type: str = Field(description="The type of the entity.")
    description: str = Field(description="Description of the entity.")
    relationships: List[str] = Field(description="Relationships of the entity.")


class TaskEvaluation(BaseModel):
    suggestions: List[str] = Field(
        description="Suggestions to improve future similar tasks."
    )
    quality: float = Field(
        description="A score from 0 to 10 evaluating on completion, quality, and overall performance, all taking into account the task description, expected output, and the result of the task."
    )
    entities: List[Entity] = Field(
        description="Entities extracted from the task output."
    )
```

> 这里有一个Converter的类，调用to_pydantic()的方法，可以调用大模型+pydantic定义，转成一个pydantic对象。支持function_calling或者直接prompt。更多细节见：`TaskEvaluator`的`evaluate`方法


### Memory项目里是怎么实现的
TODO: 看memory包

### 项目里是怎么调用llm的
litellm

### Printer类

### parser里的repair_json是?