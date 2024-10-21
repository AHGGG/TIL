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


wrapper内部, 初始化干了三个事情:
1. 通过`self.__class__.__dict__.items()`拿到crew所在类里所有的方法和属性
- `self.__class__`: 拿到crew装饰器所在的Class定义
- `.__dict__`: 拿到crew装饰器所在class内部所有的属性和方法(包括实例属性和类属性)
2. 根据标识`is_agent`和`is_task`找到所有的agents, tasks
3. 实例化所有的agent和所有的task, 放到一个数组中

### Crew类kickoff运行逻辑
```

```