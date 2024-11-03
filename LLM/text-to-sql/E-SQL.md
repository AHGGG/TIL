这篇论文在SQL中涉及到几个关键的模块: 
1. Candidate SQL Generation (CSG): 候选SQL生成
2. Candidate Predicate Generation (CPG): 与候选SQL相关联的条件与值的查询
3. Schema Filtering: 过滤出与SQL不相关的table与column
4. Question Enrichment (QE): 输入重写
5. SQL Refinement (SR): 最终的SQL选择与生成

### Candidate SQL Generation (CSG)
SQL生成的时候，注入了和当前db不相关的其他db的3个示例，按照简单到复杂排列。

注入了选中的database的描述，还有相关的列的一些数据示例（还要数据是要相关的数据）

### Candidate Predicate Generation (CPG)
解析出候选SQL中可能存在的一些状态。例如我们候选SQL中涉及到一个`name`的column, 那么通过SQL解析的库, 对于name的操作是: `= >= <= < >`这些中的一种, 还有name有哪些可选的值(通过`LIKE '%<VALUE>%'`语法来进行筛选)

这一个步骤最后筛选出来的结果是`<table>.<column> <operation> <value>`的list. 

例如: `<table>.name = "aaaa"`, `<table>.name = "bbb"`

### Schema Filtering
消除与SQL无关的数据库表和列. 只提取出相关的table和column.

这篇论文发现使用最新的LLM的时候, 使用SF会导致效果变差, 所以没有使用这个策略.

### Question Enrichment (QE)
其实就是输入重写. 将一句简单的输入重写为两部分: 
1. Enriched Input
2. Enriched Reasoning

实现方面, 通过使用few-shot的方式在prompt里注入了几个输入重写的示例, 来确保大模型会按照输入的模式进行输出.

最后这篇论文将**输入+丰富后的输入+丰富后的大模型处理过程**拼在一起放到的prompt中.(有些论文中还会进行迭代, 这篇论文暂时没有选择迭代)

TODO: 补充示例

### SQL Refinement (SR)
因为前面的候选SQL中可能有错误, 一些常用的做法: 
1. 生成多个SQL, 选择与自然语言问题最匹配、最符合逻辑的SQL(the most consistent)作为最终的结果
2. refiner agent: 检查SQL语法+是否可执行, 如果不行的话, agent会进行修复工作, 直到通过/达到最大迭代上限.

这篇文章的做法:
执行SQL, 看是否有错误. 如果有错误信息, 拿着错误信息+候选SQL+丰富后的输入, 再结合db的schema, 生成一个新的SQL

### 一些技巧
#### Null值的处理
在给llm提供每个column的示例值的时候，如果这个column包含Null值，那么提供的示例值里也要有Null（让llm注意到潜在的Null值）

#### 代码解析
通过使用`sqlglot`这个包来解析SQL, 然后拿到SQL涉及到的column的值和操作(是大于/小于/大于等于等)

## reference
https://arxiv.org/abs/2409.16751
https://github.com/HasanAlpCaferoglu/E-SQL