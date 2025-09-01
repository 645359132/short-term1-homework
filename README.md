孩子们这是一个homework

# 一、贡献指南
1.	Fork 本项目。点击 Fork 按钮，创建一个新的派生项目到自己的工作区（Create a new fork）

2.	克隆派生
```bash
# 克隆项目到本地（注意是派生项目的链接，不是原始项目）
git clone https://github.com/your-user-name/your-fork-name.git
```

3.	创建分支
```bash
# 创建并切换到本地新分支，分支的命名尽量简洁，并与想要解决的问题相关
git checkout -b your-branch-name
```

4.	修改文档内容或者新增文档

5.	提交更改 
```bash
git commit -m 'your-commit-content'
```

6.	推送到分支
```bash
git push --set-upstream origin your-branch-name
```
7.	提交合并请求
提交时添加标题和描述信息
8. 等待审核，审核通过后，合并分支到主分支
# 二、环境配置
1. 检查uv环境
```bash
uv --version
```
如果有报错，请移步 https://github.com/astral-sh/uv
2. 创建虚拟环境并安装依赖
```bash
uv sync
```
3. 没什么岔子就可以开始写代码了
