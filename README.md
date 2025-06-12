# 阻塞作业车间调度问题求解器

这是一个用于解决阻塞作业车间调度问题（Blocking Job Shop Problem）的Python项目。该项目实现了多种求解方法，包括约束规划（CP）求解器。

## 项目结构

```
.
├── main.py                 # 主程序入口
├── CPSolver.py            # 约束规划求解器
├── schedule.py            # 调度相关类
├── blocking_job_shop.py   # 阻塞作业车间问题定义
├── network.py             # 神经网络相关实现
├── memory.py              # 记忆模块实现
├── tools.py               # 工具函数
├── params.py              # 参数配置
├── requirements.txt       # 项目依赖
└── test/                  # 测试文件目录
```

## 环境要求

- Python 3.7+
- ortools
- numpy
- matplotlib

## 安装

1. 克隆仓库：
```bash
git clone [repository-url]
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行主程序：
```bash
python main.py
```

2. 运行测试：
```bash
python test.py
```

## 主要功能

- 阻塞作业车间调度问题的求解
- 约束规划（CP）求解器实现
- 神经网络辅助求解
- 调度结果可视化

## 许可证

[添加许可证信息] 