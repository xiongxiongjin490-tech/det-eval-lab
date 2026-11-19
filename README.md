# Det Eval Lab · 目标检测评估台

[English](README.en.md) · [来源说明](SOURCES.md) · [AI 辅助说明](AI_USE.md)

把标注与检测结果 CSV 变成可审查的评估报告：先校验框体合法性，再按图像与类别做 IoU 贪心匹配，输出逐类 Precision / Recall、PR 曲线点和 AP@0.5、mAP@0.5。项目独立编写，不包含检测器、模型权重、训练流程或公开数据集。

**当前演示使用程序生成的合成标注与检测框，不是任何训练模型的真实输出，不代表公开 benchmark 成绩。** 逐类结果与 PR 点都在 [results/](results/)。

## 能力与成果

- 校验缺失列、未知列、非有限坐标、`x2 <= x1` / `y2 <= y1` 非法框，以及越界检测分数。
- 按 `(image_id, class)` 分组，检测按分数降序贪心匹配，IoU 达到阈值记为 TP，其余检测记 FP，未匹配标注记 FN。
- 逐类输出标注数、检测数、TP / FP / FN、Precision、Recall 与 AP；总体给出宏平均 P / R 与 mAP。
- 导出 JSON 报告、逐类汇总 CSV、PR 点 CSV 和 Markdown 报告，记录输入 SHA-256 与运行环境。

## 快速复跑

Python 3.10+。运行时只依赖标准库；运行测试需要 pytest。

```bash
python -m pip install -e ".[dev]"
python -m det_eval_lab demo
python -m pytest
```

默认重建 `results/` 与网页读取的 `docs/data.json`。CI 可用独立目录：

```bash
python -m det_eval_lab demo --quick --output results-quick --docs results-quick/data.json
```

## 评估自己的结果

```bash
python -m det_eval_lab eval --gt annotations.csv --det detections.csv --output my-report
```

标注 CSV 必需列：`image_id,class,x1,y1,x2,y2`；检测 CSV 额外需要 `score`（取值 0~1）。坐标为 `x1,y1,x2,y2` 像素口径，工具不感知图像分辨率与类别语义。

```csv
image_id,class,x1,y1,x2,y2
img1,bolt,10,10,50,50
```

```csv
image_id,class,x1,y1,x2,y2,score
img1,bolt,11,11,49,49,0.92
```

默认 IoU 阈值 0.5，可用 `--iou` 调整。外部输入默认为 `external-unverified`，来源标签是用户声明，不是认证。CLI 在输入无效时返回退出码 2。

## 如何解读结果

`results/report.md` 是报告入口，`per_class.csv` 保留每类分母与指标，`pr_points.csv` 保留逐检测分数对应的 Recall / Precision。AP 采用全点插值包络面积；同一类内检测分数相同时按输入顺序处理。本工具只做离线结果统计，不判断框体标注质量、类别合理性或模型泛化能力；不同数据集、不同分辨率口径的结果不能直接横向比较。

## 项目边界

本项目受目标检测离线评估的通用需求启发，实际实现是独立的 IoU / PR / AP 统计工具；**没有复现任何检测器训练流程，也不绑定具体网络结构**。只引用公开评估约定与标准库文档，未复制上游代码、权重或数据集。见 [SOURCES.md](SOURCES.md)。代码采用 MIT 许可；引用资料保留其原有权利。

## 固定环境与验证

使用 Python 3.12。锁文件包含开发、构建及跨平台运行依赖，首次安装后即可复跑：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q
python -m det_eval_lab demo
```

更新演示数据后，重新执行 `python -m det_eval_lab demo` 即可重建报告。
