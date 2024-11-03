这篇论文提出的框架，有三个组成部分：entity和context的召回，schema selection，SQL生成

## 处理流程
1. keyword extraction
2. entity retrieval
3. context retrieval
4. column filtering
5. table selection
6. column selection
7. candidate generate
8. revision

### keyword extraction
没啥，就是直接调用一个few-shot的prompt，拿到提取出来的一些关键词
```python
Objective: Analyze the given question and hint to identify and extract keywords, keyphrases, and named entities. These elements are crucial for understanding the core components of the inquiry and the guidance provided. This process involves recognizing and isolating significant terms and phrases that could be instrumental in formulating searches or queries related to the posed question.

Instructions:

Read the Question Carefully: Understand the primary focus and specific details of the question. Look for any named entities (such as organizations, locations, etc.), technical terms, and other phrases that encapsulate important aspects of the inquiry.

Analyze the Hint: The hint is designed to direct attention toward certain elements relevant to answering the question. Extract any keywords, phrases, or named entities that could provide further clarity or direction in formulating an answer.

List Keyphrases and Entities: Combine your findings from both the question and the hint into a single Python list. This list should contain:

Keywords: Single words that capture essential aspects of the question or hint.
Keyphrases: Short phrases or named entities that represent specific concepts, locations, organizations, or other significant details.
Ensure to maintain the original phrasing or terminology used in the question and hint.

Example 1:
Question: "What is the annual revenue of Acme Corp in the United States for 2022?"
Hint: "Focus on financial reports and U.S. market performance for the fiscal year 2022."

["annual revenue", "Acme Corp", "United States", "2022", "financial reports", "U.S. market performance", "fiscal year"]

Example 2:
Question: "In the Winter and Summer Olympics of 1988, which game has the most number of competitors? Find the difference of the number of competitors between the two games."
Hint: "the most number of competitors refer to MAX(COUNT(person_id)); SUBTRACT(COUNT(person_id where games_name = '1988 Summer'), COUNT(person_id where games_name = '1988 Winter'));"

["Winter Olympics", "Summer Olympics", "1988", "1988 Summer", "Summer", "1988 Winter", "Winter", "number of competitors", "difference", "MAX(COUNT(person_id))", "games_name", "person_id"]

Example 3:
Question: "How many Men's 200 Metres Freestyle events did Ian James Thorpe compete in?"
Hint: "Men's 200 Metres Freestyle events refer to event_name = 'Swimming Men''s 200 metres Freestyle'; events compete in refers to event_id;"

["Swimming Men's 200 metres Freestyle", "Ian James Thorpe", "Ian", "James", "Thorpe", "compete in", "event_name", "event_id"]

Task:
Given the following question and hint, identify and list all relevant keywords, keyphrases, and named entities.

Question: {QUESTION}
Hint: {HINT}

Please provide your findings as a Python list, capturing the essence of both the question and hint through the identified terms and phrases. 
Only output the Python list, no explanations needed. 

```
### entity retrieval
根据前面的keywords，查询到相似的columns和values

### context retrieval
通过预处理的时候，将一些描述信息向量化到vector db中。然后这里就拿着提取出来的关键词去里面查询最相似的哪些描述信息。

> 预处理的时候使用到了局部敏感哈希（Locality Sensitive Hashing, LSH）的技术，用来检索数据库中与关键字最相似的值。预处理的时候拿到db中不一样的值建立索引。

> 局部敏感哈希（Locality Sensitive Hashing, LSH）是一种算法，用于将相似的对象映射到相同或相似的哈希桶中，从而使得在高维空间中进行相似性搜索更加高效。LSH的关键思想是设计哈希函数，使得相似的输入（如相似的向量或集合）以高概率被哈希到相同的桶中，而不相似的输入则以低概率被哈希到同一桶中 ---- chatgpt
> 内部实现用到了MinHash（处理大规模数据集的相似性查询）


### column filtering
遍历每个table的每个column信息，拿着每个column的信息，调用llm，输出一个is_column_information_relevant，为Yes/No

如果llm给出了了yes，那么`tentative_schema[table_name].append(column_name)`，记录table和column的信息
```python
{{
  "chain_of_thought_reasoning": "One line explanation of why or why not the column information is relevant to the question and the hint.",
  "is_column_information_relevant": "Yes" or "No"
}}
```

<div></div>

### table selction
这一部分prompt注入的DB_SCHEMA是前面column filtering拿到的tentative_schema，也就是筛选后的schema，不是全量的schema定义。
table selection返回的结构如下：
```python
{{
  "chain_of_thought_reasoning": "Explanation of the logical analysis that led to the selection of the tables.",
  "table_names": ["Table1", "Table2", "Table3", ...]
}}
```

### column selection
同样，这里注入的也是部分DB_SCHEMA
再对column进行过滤。先让llm一次性基于上下文，给出关联table+column，会给出多个table，每个table都输出有关系的column：
```python
{{
  "chain_of_thought_reasoning": "Your reasoning for selecting the columns, be concise and clear.",
  "table_name1": ["column1", "column2", ...],
  "table_name2": ["column1", "column2", ...],
  ...
}}
```

### candidate generate
提供最小范围的schema+相关的column值+召回的相关描述，生成一个SQL

执行这个SQL，拿到结果集。（代码里其实只生成了一个，执行了一个）
- 如果执行正确，将执行结果拿到变成一个set，作为`clusters`的key，value则是本次执行的SQL。后面如果存在同样的key，那么新的SQL继续append的结果计算出来的key对应的value数组中。
	- 最后找到`clusters`中最大的聚类（即包含最多 SQL 查询的列表），然后从中选择最短的 SQL 查询作为最后的结果返回。
- 如果没有一个SQL有执行结果，那么直接返回第一个SQL
```python
def aggregate_sqls(db_path: str, sqls: List[str]) -> str:
    """
    Aggregates multiple SQL queries by validating them and clustering based on result sets.
    
    Args:
        db_path (str): The path to the database file.
        sqls (List[str]): A list of SQL queries to aggregate.
        
    Returns:
        str: The shortest SQL query from the largest cluster of equivalent queries.
    """
    # 执行多个SQL
    results = [validate_sql_query(db_path, sql) for sql in sqls]
    clusters = {}

    # Group queries by unique result sets
    for result in results:
        if result['STATUS'] == 'OK':
            # Using a frozenset as the key to handle unhashable types like lists
            # frozenset可以当作不能被改变的set。注意这里将sql结果的每一行都转成了tuple，set中放的是一个个的tuple，然后这个frozenset可以作为dict的key
            key = frozenset(tuple(row) for row in result['RESULT'])
            if key in clusters:
	            # 把执行结果一样的放到一起
                clusters[key].append(result['SQL'])
            else:
                clusters[key] = [result['SQL']]

	# if {} ==> 注意if后面跟一个空的dict，返回的是False
    if clusters:
        # Find the largest cluster（找到SQL最多的cluster）
        largest_cluster = max(clusters.values(), key=len, default=[])
        # Select the shortest SQL query from the largest cluster
        if largest_cluster:
	        # 找到长度最小的SQL
            return min(largest_cluster, key=len)

	# 如果clusters中没有东西，那么返回第一个SQL
    logging.warning("No valid SQL clusters found. Returning the first SQL query.")
    return sqls[0]

```

### revision
1. 拿到前一步的SQL，先执行一遍，拿到结果。
2. 解析SQL，拿到Dictionary of tables and their columns with condition literals，结构如下：`Dict[str, Dict[str, List[str]]]`。key是table的名字，value是一个dict（key是column名字，value是column可选一些值）
- 主要就是判断SQL中涉及到的value在这个table.column中是否存在。如果不存在，那么找到最相似的（默认设置的0.4），然后作为额外的信息放到prompt中，给llm说你使用的value不存在，存在的是xxxx。

3. 然后就是走prompt，调用llm，拿到修改后的SQL。
4. 最后再走一遍执行，聚类，选择一个最好的SQL。
5. 得到最后的SQL


## reference
https://arxiv.org/abs/2405.16755
https://github.com/ShayanTalaei/CHESS