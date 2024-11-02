https://astral.sh/blog/uv-unified-python-packaging
https://docs.astral.sh/uv/reference/cli/#uv


### 命令
uv lock: 更新项目中的lock文件。如果uv.lock不存在，则将创建它。如果存在，则会使用存在的lock文件

uv sync: Syncing ensures that all project dependencies are installed and up-to-date with the lockfile.

uv venv: 在当前目录创建一个虚拟环境

uv pip sync ./requirement.txt: 基于一个requirement依赖文件安装依赖
- 太快了！！！爽