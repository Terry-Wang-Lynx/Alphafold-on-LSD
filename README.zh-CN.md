[English](README.md) | [简体中文](README.zh-CN.md)

离线 HTML 文档（`docs/`，建议在本地浏览器中打开）：[概览](docs/index.zh-CN.html) · [安装](docs/install.html) · [使用](docs/guide.html) · [API](docs/api.html) · [数值计算](docs/numerics.html) · [迁移](docs/migration.html) · [来源与归属](docs/provenance.html) · [参与贡献](docs/contributing.html)

本页为中文版主页；安装、使用、API、数值计算等详细文档目前为英文。

AlphaFold on LSD · `0.3.0rc3` · 候选发布版  
发行包名：`hallucination-lab` · Python 导入包：`hallucination_lab` · 命令行工具：`hallulab`

# 可自定义的蛋白质 hallucination。

通过普通 Python 脚本，自定义可微优化目标、选择 AF2 或 AF3（或组合多条件目标），并精确控制序列优化过程。

框架提供按需导入的模型适配器、显式 distogram 数学目标函数、采用 float64 宿主侧动量计算的可替换优化步骤、分阶段控制渐变、原子写入且无 pickle 的检查点、固定输入数值审计门控以及明确的评估记录。模型运行环境与权重需要单独安装。

## 从一个数值示例开始

```sh
python3 -m venv .venv
.venv/bin/python -m pip install '.[jax]'
.venv/bin/hallulab doctor
.venv/bin/hallulab run hallucination_lab.toy:build --max-steps 3
.venv/bin/hallulab run hallucination_lab.toy:build --resume runs/toy/checkpoint.npz
```

请在源码目录中执行上述命令。`toy` 使用人工构造的二次目标函数，并非蛋白质结构预测模型。如果安装 wheel，请将 wheel 文件路径传给 `pip`，并在同一环境中安装 JAX。

## 可以自定义什么

| 层级 | 公开接口 |
|----|----|
| 模型 | `AF2`（ColabDesign 适配器）、`AF3`（适配后的 trunk/distogram 路径及前向结构采样）、`Functional`（自定义可微后端）与 `Composite`（显式残基对齐的多条件链式法则组合）。 |
| 目标函数 | `distribution`（掩码交叉熵）、`interval`（距离区间事件概率或排斥）、`contact`（全对或每行 top-k 接触）、支持控制参数覆写权重的 `weighted` 组合，或任意返回标量 / `(scalar, metrics)` 的自定义 JAX 函数。 |
| 优化 | `Run`、`Stage` 控制与学习率渐变、`gradient_step`（FP64 宿主侧 Adam/SGD 动量、梯度裁剪、自定义梯度变换）、`mutation_step`（无梯度单点突变提议）、按阶段分派优化步骤与早停观察器。 |
| 验证与记录 | `audit_backend` / `compare_evaluations` 数值一致性门控、`paired_evaluate` 固定条件对比评估、`preflight`（`hallulab doctor`）、源码与运行环境指纹、逐步骤确定性随机密钥、原子无 pickle 检查点与显式导出（`save_prediction`、`write_fasta`）。 |

## 组合目标与扩展工作流

显式目标函数会替代上游模型隐藏的默认损失；调用者需自行声明所有期望项：

```python
from hallucination_lab.losses import distribution, interval, weighted

objective = weighted({
    "shape": (1.0, distribution(target, pair_mask, edges=edges)),
    "range": (0.2, interval(other_mask, edges, lower=4.0, upper=12.0)),
})
```

完整脚本示例见 [examples/af2.py](examples/af2.py)、[examples/af3.py](examples/af3.py)、[examples/stages.py](examples/stages.py) 与 [examples/customize.py](examples/customize.py)；AF2/AF3 的 Linux/CUDA 固定环境配置见 [docs/install.html](docs/install.html)。

## 仓库结构与 CPU 验证

- `src/hallucination_lab/`：核心优化引擎、序列松弛、distogram 损失、数值门控、成对评估、溯源与诊断工具，以及按需导入的 `AF2`、`AF3`、`Composite` 和 `Functional` 后端。
- `examples/`：独立的 CLI 工厂脚本（`toy.py`、`af2.py`、`af3.py`、`stages.py`、`customize.py`、`make_af3_input.py`）与 JSON 配置模板。
- `environments/`：固定版本的 Linux/CUDA 参考环境配置与源码校验脚本（`verify_af2_source.py`）。
- `tests/`：纯 CPU 测试套件，涵盖解析 float64 损失与梯度参考验证、80 位精度 `Decimal` 优化器算术验证、AF2/AF3 适配器契约、数值门控、检查点/轨迹恢复以及发布文件白名单检查。
- `docs/`：自包含离线 HTML 文档（安装、使用指南、API 参考、数值约定、迁移、来源归属与贡献指南）。
- `scripts/audit_release.py` 与 `release-files.json`：显式公开文件白名单与源码包完整性审计脚本。

无需 GPU 或模型权重，即可运行完整的 CPU 测试、代码检查与发布合规审计：

```sh
.venv/bin/python -m pip install '.[dev,jax]'
.venv/bin/python -m ruff check src tests examples scripts
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit_release.py
```

## 计算范围

AF2 使用单独安装的 ColabDesign 运行环境，仅对最后一次 recycle 求导，并默认开启同状态前向标量守卫。AF3 通过适配后的 trunk 与 distogram 路径求导，使用重新特征化的离散原子载体与软 query 特征（默认 `profile_mode="auto"` 在存在同源序列时保留原生 MSA profile）。AF3 的 diffusion 坐标和置信度只用于前向计算；验证锁定序列时请使用 `AF3.predict_hard_sequence()` 以原生离散输入重新特征化。离散序列优化步骤使用明确声明的直通代理梯度。

本工具提供数学目标函数。目标函数值降低本身不能证明折叠、亲和力或生物学功能。更改运行环境或梯度语义前，请阅读 [数值计算约定（英文）](docs/numerics.html)。

## 发行包内容与许可

wheel 包含可导入的库（`hallucination_lab`）和命令行工具（`hallulab`）。源码包还包含离线 HTML 文档、示例、运行环境配置、软件测试与发布审计脚本。不随包提供模型权重、数据库或项目专用的设计数据集。

独立科研软件。项目代码采用 [Apache-2.0](LICENSE) 许可，上游归属与声明见 [NOTICE](NOTICE)。外部运行环境、模型参数及输出仍受各自条款约束。
