[English](README.md) | [简体中文](README.zh-CN.md)

[概览](docs/index.zh-CN.html) · [安装](docs/install.html) · [使用](docs/guide.html) · [API](docs/api.html) · [数值计算](docs/numerics.html) · [迁移](docs/migration.html) · [来源与归属](docs/provenance.html) · [参与贡献](docs/contributing.html)

本页为中文版主页；安装、使用、API 等详细文档目前为英文。

AlphaFold on LSD · 0.3.0rc3 · 候选发布版

# 可自定义的蛋白质 hallucination。

通过普通 Python 脚本，自定义优化目标、选择 AF2 或 AF3，并控制序列优化过程。

框架提供模型适配器、数学目标函数、可替换的优化步骤、分阶段控制、检查点和明确的评估记录。模型运行环境与权重需要单独安装。

## 从一个数值示例开始

```sh
python3 -m venv .venv
.venv/bin/python -m pip install '.[jax]'
.venv/bin/hallulab doctor
.venv/bin/hallulab run hallucination_lab.toy:build --max-steps 3
.venv/bin/hallulab run hallucination_lab.toy:build --resume runs/toy/checkpoint.npz
```

请在解压后的源码目录中执行这些命令。toy 使用人工构造的二次目标函数，并非蛋白质预测模型。如果安装 wheel，请将 wheel 文件路径传给 pip，并在同一环境中安装 JAX。

## 可以自定义什么

| 层级 | 公开接口 |
|----|----|
| 模型 | AF2、AF3、Functional 和 Composite 后端。 |
| 目标函数 | 分布交叉熵、距离事件、排斥、接触、加权组合，或自定义的 JAX 可调用对象。 |
| 优化 | 阶段控制与参数渐变、Adam 或 SGD、梯度变换，以及调用者自定义的优化步骤和观察器。 |
| 记录 | 源码与运行环境身份、随机密钥、控制参数、被评估序列的身份、原子写入的检查点和显式导出。 |

## 计算范围

AF2 使用单独安装的 ColabDesign 运行环境。AF3 通过适配后的 trunk 与 distogram 路径求导；diffusion 坐标和置信度只用于前向计算。离散序列优化步骤使用明确声明的直通代理梯度。

本工具提供数学目标函数。目标函数值降低本身不能证明折叠、亲和力或生物学功能。更改运行环境或梯度语义前，请阅读[数值计算约定（英文）](docs/numerics.html)。

## 发行包内容

wheel 包含可导入的库和命令行工具。源码包还包含 HTML 文档、示例、运行环境配置、软件测试与构建检查。不随包提供模型权重、数据库或项目专用的设计数据集。

独立科研软件。项目代码采用 Apache-2.0 许可。外部运行环境、模型参数及输出仍受各自条款约束。
