# PepMamba 论文最终结果来源审计

审计日期：2026-07-24
项目目录：`.`
配套逐行表：[`PAPER_RESULT_PROVENANCE_TABLE.csv`](./PAPER_RESULT_PROVENANCE_TABLE.csv)
既有数据审计：[`DATA_SPLIT_AND_THRESHOLD_AUDIT.md`](./DATA_SPLIT_AND_THRESHOLD_AUDIT.md)

## 0. 审计边界和总判定

本次没有修改旧代码、数据、checkpoint、日志或历史结果，只新增本报告和配套 CSV。
结论不是根据目录名推测，而是交叉检查了结果 JSON、run config、checkpoint
metadata、validation/test 预测、阈值扫描、汇总表和生成脚本。

风险定义严格采用任务要求：

- **GREEN**：超参数、checkpoint 和阈值均只由 validation 确定，test 仅评价一次。
- **YELLOW**：阈值来自 validation，但 test 曾被重复查看、最终选择依据无法确认，
  或关键来源文件缺失。
- **RED**：test 用于阈值、超参数、trial 或最终运行选择。

### 总结

1. **当前能确认的论文候选内部结果中，没有一条满足严格 GREEN。**
2. `BITENET_FINAL_KEEP` 是明确的 **RED**：trial 由 test 指标排名推荐，最终
   `0.605` 阈值也来自 test 标签扫描。
3. 四组论文 BITENET 消融虽然各自使用 validation 选 checkpoint 和阈值，
   但共同使用了上述由 test 选择出的 trial-2 超参数，因此按给定规则仍是
   **RED**。
4. 最新主干总结采用的 BioMamba `AUROC=0.8574` 是五次相同 seed 重跑中的
   指定一次运行；没有 validation-only 的“选择哪次运行”记录，而且该运行的
   test AUROC 是五次中最高者。再加上其超参数本身来自 test 排名，判为
   **RED**。
5. INTERPEP、PEPBIND、PEPNN 目前只有 checkpoint 和 validation metadata，
   没有保存的 test 结果或 validation/test 预测，超参数选择历史也不完整，
   判为 **YELLOW / 暂不可作为论文最终结果**。
6. 项目中没有找到 PepCA、PepNN、PepBind、InterPep 等外部方法的论文对比结果表
   或逐值来源，因此当前无法审计这些对比数字。

## 1. 目前最可能进入论文的结果

### 1.1 明确标记为论文最终消融的目录

[`PAPER_ABLATION_FINAL_4GROUPS/README.md:1-10`](./PAPER_ABLATION_FINAL_4GROUPS/README.md#L1-L10)
明确称其为“四个最终 BITENET 消融”，并规定主文只使用 seed 42；
[`README.md:37-48`](./PAPER_ABLATION_FINAL_4GROUPS/README.md#L37-L48)
又将 `seed42_results/` 称为“final ... for paper use”。

四组来源由
[`SOURCE_PATHS.txt:1-15`](./PAPER_ABLATION_FINAL_4GROUPS/SOURCE_PATHS.txt#L1-L15)
明确记录：

| 论文组 | 原始运行目录 | seed=42 行数 |
|---|---|---:|
| ESM-2 表征消融 | `tuning_results/bitenet_ablation_pretrained_repr_final_20260619_183044` | 4 |
| protein prior / interaction-refined 头消融 | `tuning_results/bitenet_ablation_branch_design_multiseed_20260620_033915` | 3 |
| Cross-first / Mamba-first 顺序消融 | `tuning_results/bitenet_ablation_cross_first_mamba_fusion_multiseed_20260630_161351` | 2 |
| Transformer/Mamba1/Mamba2/Mamba3/BioMamba | `tuning_results/bitenet_ablation_backbone_multiseed_20260630_200302` | 5 |

[`SEED42_RESULTS_SUMMARY.csv:2-15`](./PAPER_ABLATION_FINAL_4GROUPS/SEED42_RESULTS_SUMMARY.csv#L2-L15)
共 14 行。逐行将汇总值与其 `source_run/result.json` 比较，所有指标差异均小于
`5e-7`，即只是六位小数舍入，不是人工改写。14 个源目录也都同时存在：

- `result.json`
- `best_state.pth` 或 `best_checkpoint.pth`
- `validation_predictions.npz`
- `test_predictions.npz`
- `threshold_scan.csv`

这些文件完整性足以重算指标，但不能消除上游超参数的 test 选择污染。

### 1.2 BITENET_FINAL_KEEP

[`BITENET_FINAL_KEEP/README.md:3-13`](./tuning_results/BITENET_FINAL_KEEP/README.md#L3-L13)
称该目录为 canonical selected result，来源是 full-search trial 2；
[`README.md:15-30`](./tuning_results/BITENET_FINAL_KEEP/README.md#L15-L30)
给出正式 test 指标，并明确写明 `0.605` 来自 **test threshold scan**。

正式行见
[`final_metrics.csv:2-9`](./tuning_results/BITENET_FINAL_KEEP/final_metrics.csv#L2-L9)：

| AUROC | AUPRC | MCC | F1 | Precision | Sensitivity | Specificity | Threshold |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.856357 | 0.396733 | 0.386015 | 0.417311 | 0.444371 | 0.393357 | 0.971050 | 0.605 |

F1 由保存的
[`trial2_threshold_scan.csv`](./tuning_results/BITENET_FINAL_KEEP/threshold_scan/trial2_threshold_scan.csv)
中 `threshold=0.605` 行直接取得。

### 1.3 最新 BioMamba 主干主表

[`mamba_backbone_ablation_summary_20260724.md:50-62`](./tuning_results/mamba_backbone_ablation_summary_20260724.md#L50-L62)
明确说主表采用“原环境重跑”并报告：

| AUROC | AUPRC | MCC | F1 | Precision | Sensitivity | Specificity | Threshold |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.857393 | 0.397232 | 0.376227 | 0.411530 | 0.403244 | 0.420163 | 0.963401 | 0.56 |

精确来源是
[`biomamba_rerun_original_env_20260722/.../result.json:31-54`](./tuning_results/biomamba_rerun_original_env_20260722/seed_42/pep_mamba_biomamba_ours/result.json#L31-L54)。
该目录有 checkpoint、validation/test 预测和 validation threshold scan。

但是项目中另有 `repeat2`、`repeat3`、`repeat4`、`repeat5` 四次相同 seed=42
运行。五次 test AUROC 分别为：

| 运行 | Test AUROC | Test AUPRC | Test MCC | Validation AUROC |
|---|---:|---:|---:|---:|
| 原环境首次重跑 | **0.857393** | 0.397232 | 0.376227 | 0.889128 |
| repeat2 | 0.856617 | 0.396178 | 0.385520 | 0.888961 |
| repeat3 | 0.856540 | 0.394273 | 0.380086 | 0.887975 |
| repeat4 | 0.856940 | **0.397568** | **0.386732** | 0.888925 |
| repeat5 | 0.856138 | 0.395878 | 0.383503 | 0.888323 |

首次重跑同时是这五次中的 validation AUROC 和 test AUROC 最高者，因此事后
采用它与“按 validation AUROC 排名”并不矛盾；但报告
[`64-80`](./tuning_results/mamba_backbone_ablation_summary_20260724.md#L64-L80)
没有保存任何预先声明或 validation-only 的运行选择 manifest，而 test 已被
重复评价五次。因此，仅就“五次中选哪一次”最多只能判为 YELLOW；该行整体仍因
共同超参数来自 test-ranked trial 2 而判为 RED。

### 1.4 四数据集主模型

当前项目没有一张包含四数据集最终 test 指标和来源的主结果表。可找到的材料为：

| 数据集 | checkpoint | validation metadata | test 结果/预测 | 判定 |
|---|---|---|---|---|
| BITENET / TS125 | 多套，当前 config 指向目录不存在 | 多套 | 多套、互相冲突 | RED/尚未确定唯一主结果 |
| INTERPEP / TS251 | `out_model/interpep/biomamba_interpep_seed42/...pth` | 有 | 无 | YELLOW |
| PEPBIND / TS639 | `out_model/pepbind/biomamba_pepbind_seed42/...pth` | 有 | 无 | YELLOW |
| PEPNN / TS092 | `out_model/pepnn/biomamba_pepnn_seed42/...pth` | 有 | 无 | YELLOW |

三个 metadata 中保存的 validation 信息：

| 数据集 | Best epoch | Val AUROC | Val AUPRC | Val MCC threshold |
|---|---:|---:|---:|---:|
| interpep | 16 | 0.822164 | 0.485255 | 0.53 |
| pepbind | 21 | 0.852810 | 0.415264 | 0.81 |
| pepnn | 24 | 0.851852 | 0.476027 | 0.43 |

这些 `.pt` metadata 是二进制文件，没有文本行号；其字段与
[`train.py:280-293`](./train.py#L280-L293) 的保存结构一致。目录中没有
validation/test prediction 文件，也没有 test metrics 文件，因而目前无法列出
论文要求的七项 test 指标。

当前 [`config.py:51-52`](./config.py#L51-L52) 的 `EXP_NAME` 是
`biomamba_bitenet_seed42`，但 `out_model/bitenet/biomamba_bitenet_seed42/`
实际不存在；因此当前 `test.py -d bitenet` 也不能直接定位上述论文主表的
BioMamba checkpoint。

### 1.5 外部方法对比

项目 README 只有运行说明，
[`README.md:18-33`](./README.md#L18-L33) 没有结果表。对项目内 Markdown、
JSON、TXT 和非预测 CSV 的检索未找到 PepCA、PepNN、PepBind、InterPep
对比数值及其论文表号、版本或评估脚本。

结论：外部方法数字若已存在于论文草稿之外，本项目无法为其提供 provenance。
它们应逐项从原论文/官方实现导入，并记录“原文报告”还是“本项目复现”，不能
与本项目生成的数字混写。

## 2. test 是否参与选择：完整因果链

### 2.1 BITENET trial 2 的 checkpoint 是 validation 选择

full-search runner 在
[`tune_bitenet_extreme.py:589-599`](./tuning_results/BITENET_FINAL_KEEP/scripts/tune_bitenet_extreme.py#L589-L599)
按 `(val_auc, val_auprc)` 保存 checkpoint，然后以该 validation threshold
评价 test。单个 epoch/checkpoint 的选择本身合规。

### 2.2 但 trial 和超参数明确由 test 排名选择

同一脚本：

- [`214-224`](./tuning_results/BITENET_FINAL_KEEP/scripts/tune_bitenet_extreme.py#L214-L224)
  直接用 test sensitivity、specificity、precision、MCC、AUROC 计算达标和 gap；
- [`349-372`](./tuning_results/BITENET_FINAL_KEEP/scripts/tune_bitenet_extreme.py#L349-L372)
  排名键包含 test MCC、AUROC、precision；
- [`379-385`](./tuning_results/BITENET_FINAL_KEEP/scripts/tune_bitenet_extreme.py#L379-L385)
  把排名写入 `test_ranking`，并把第一名参数写入 `recommended_params`；
- [`409-438`](./tuning_results/BITENET_FINAL_KEEP/scripts/tune_bitenet_extreme.py#L409-L438)
  生成 Test Ranking，并明确写出“Recommend trial ... based on ... test ranking”。

因此 trial 2 的超参数选择是明确 **RED**，即使 trial 内阈值原先是 validation
MCC 阈值 `0.63`。

### 2.3 最终 0.605 又被 test threshold scan 替换

trial 2 原始结果
[`trial2_source/result.json:42-62`](./tuning_results/BITENET_FINAL_KEEP/trial2_source/result.json#L42-L62)
记录 validation 阈值 `0.63`，test 也固定使用 `0.63`。但
[`BITENET_FINAL_KEEP/README.md:28-30`](./tuning_results/BITENET_FINAL_KEEP/README.md#L28-L30)
明确将最终值改成 test scan 选出的 `0.605`。

所以 `BITENET_FINAL_KEEP/final_metrics.csv` 不是“validation 阈值固定评价”的结果。

### 2.4 四组消融继承了被 test 选择的超参数

统一消融 runner 的
[`FINAL_PARAMS`](./run_bitenet_pretrained_repr_ablation.py#L46-L60)
与 trial 2 的
[`params`](./tuning_results/BITENET_FINAL_KEEP/trial2_source/result.json#L5-L17)
逐项一致。

runner 本身是：

- checkpoint：validation AUROC，AUPRC tie-break，
  [`616-624`](./run_bitenet_pretrained_repr_ablation.py#L616-L624)；
- 阈值：validation prediction 上做 MCC scan，
  [`626-641`](./run_bitenet_pretrained_repr_ablation.py#L626-L641)；
- test：固定使用选出的 validation threshold，
  [`636-661`](./run_bitenet_pretrained_repr_ablation.py#L636-L661)。

所以局部评估协议正确，但超参数因果链不正确。按任务规则“超参数根据 test
挑选仍判定受到 test 选择影响”，14 行均为 **RED**。

此外，各包装脚本还按 test 结果排序并写“recommended paper table”：

- protein prior/interaction-refined：
  [`run_bitenet_branch_design_ablation.py:151-192`](./run_bitenet_branch_design_ablation.py#L151-L192)
  和 [`273`](./run_bitenet_branch_design_ablation.py#L273)；
- Cross/Mamba 顺序：
  [`run_bitenet_cross_first_mamba_fusion_ablation.py:283-322`](./run_bitenet_cross_first_mamba_fusion_ablation.py#L283-L322)
  和 [`405`](./run_bitenet_cross_first_mamba_fusion_ablation.py#L405)；
- backbone：
  [`run_bitenet_backbone_ablation.py:238-264`](./run_bitenet_backbone_ablation.py#L238-L264)。

用于描述消融差异本身可以查看 test；但若据此决定“最终结构”，则又构成模型
选择。项目没有保存独立的、在查看 test 前冻结架构的 protocol。

## 3. 已发现问题逐项核实

### 3.1 通用 test.py 的 0.65

确认仍存在：

- [`test.py:126-130`](./test.py#L126-L130) 读取 `meta`，但随后硬编码
  `threshold = 0.65`；
- 第 130 行又把来源字符串覆盖为 `"validation-selected threshold"`；
- `meta["best_threshold"]` 从未被读取。

所以通用 `test.py` 打印的分类指标不能证明来自 validation 阈值。

### 3.2 test.py 的 oracle_thresh 是否进入正式文件

[`test.py:135-152`](./test.py#L135-L152) 用 test 标签搜索 `oracle_thresh`，
但脚本只 `print`，没有任何 JSON/CSV/metadata 写入语句。项目内也没有该脚本
产生的 test log。因此：

- oracle scan **存在**；
- 在当前可见代码中仅为 console debug；
- 没有证据表明它被该脚本自动写入正式结果；
- 也没有可用日志证明人工是否曾复制其数值。

这与 `BITENET_FINAL_KEEP` 不同：后者有文字和 CSV 明确证明 test scan 值进入
final metrics。

### 3.3 tune_pepbind_extreme.py

确认全部成立：

- [`464-483`](./tune_pepbind_extreme.py#L464-L483)：同时扫描 validation 和
  test 标签阈值；
- [`267-310`](./tune_pepbind_extreme.py#L267-L310)：flatten 和 rank 使用
  test 阈值下的 MCC、AUROC、precision、sensitivity、specificity；
- [`514-550`](./tune_pepbind_extreme.py#L514-L550)：按 test 排名选择第一名，
  将 `best["test"]["threshold"]` 写入推荐配置；
- [`568-573`](./tune_pepbind_extreme.py#L568-L573)：报告明说最终 trial 排名
  使用 balanced test scan。

full run 的
[`pepbind_tuning_report.md:22-56`](./tuning_results/pepbind_extreme/pepbind_full_seed42_extreme_20260615/pepbind_tuning_report.md#L22-L56)
推荐 trial 11 和 test 阈值 `0.505`。判为 **RED**。

### 3.4 refine_bitenet_single_auc860.py

确认全部成立：

- [`293-318`](./tuning_results/BITENET_FINAL_KEEP/scripts/refine_bitenet_single_auc860.py#L293-L318)
  收集 test 标签、在 test 上扫描阈值，并把选中值写入 `chosen_test`；
- [`126-135`](./tuning_results/BITENET_FINAL_KEEP/scripts/refine_bitenet_single_auc860.py#L126-L135)
  test 阈值排名包含 MCC、precision、specificity、AUROC；
- [`171-181`](./tuning_results/BITENET_FINAL_KEEP/scripts/refine_bitenet_single_auc860.py#L171-L181)
  又按 test AUROC 排 trial，并写 `test_auc_ranking`。

该 refinement 是 **RED**，不能作为论文正式选择依据。

### 3.5 BITENET_FINAL_KEEP 是否人工从多次 test 中筛选

是。其 README
[`55-62`](./tuning_results/BITENET_FINAL_KEEP/README.md#L55-L62)
说明 19 个未选 full-search trial 和两个未选 refine trial 被移入 archive；
full-search 代码又明确按 test 排名推荐。这不是单纯“保存最好 validation
checkpoint”，而是多次 test evaluation 后保留 trial 2。

### 3.6 biomamba_original_same_seed_three_runs_20260723

三次结果的参数完全一致，分别见三个 `result.json:6-19`，它们就是
BITENET trial 2 参数。每次内部均按 validation checkpoint/threshold：

- run 1：[`result.json:31-54`](./tuning_results/biomamba_original_same_seed_three_runs_20260723/run_1/biomamba_original/result.json#L31-L54)
- run 2：[`result.json:31-54`](./tuning_results/biomamba_original_same_seed_three_runs_20260723/run_2/biomamba_original/result.json#L31-L54)
- run 3：[`result.json:31-54`](./tuning_results/biomamba_original_same_seed_three_runs_20260723/run_3/biomamba_original/result.json#L31-L54)

但是：

- 没有 `selected_by_validation.json`；
- 没有三次运行的最终选择/均值文件；
- [`config.py:4-6`](./config.py#L4-L6) 只用注释称它是参数来源，实际上该目录
  没有参数搜索过程；
- 数值可追溯到早先由 test 排名选出的 trial 2。

结论：三次运行没有生成新的 BEST_PARAMS；它们是在复跑 test-selected
BEST_PARAMS。若从三次中挑一条 test 更好的结果，则再次构成 test selection。

### 3.7 四数据集阈值是否都有 validation MCC 扫描证据

不是。

- BITENET 14 条消融：有 validation predictions 和完整 `threshold_scan.csv`。
- 最新 BioMamba 重跑：有 validation predictions 和 scan。
- `BITENET_FINAL_KEEP` 原 trial-2 阈值 `0.63` 有 validation 证据，但论文归档
  的 `0.605` 明确来自 test。
- INTERPEP/PEPBIND/PEPNN：只有 metadata 中的单个 `best_threshold`，没有
  保存 validation labels/scores 或 scan 表，因此只能证明训练代码曾写入该值，
  不能独立复算。

## 4. 论文拟采用参数与实际运行

### 4.1 BITENET 和 14 条消融

BITENET 列出的参数和模型结构与实际运行一致。证据：

- 参数：
  [`run_bitenet_pretrained_repr_ablation.py:46-60`](./run_bitenet_pretrained_repr_ablation.py#L46-L60)
- 模型结构：
  [`62-69`](./run_bitenet_pretrained_repr_ablation.py#L62-L69)
- 原环境重跑的 run config：
  [`run_config.json:12-34`](./tuning_results/biomamba_rerun_original_env_20260722/run_config.json#L12-L34)

即 `n_layers=3, d_model=128, n_head=4, d_k=d_v=32, d_inner=256`，
epochs=35，seed=42。**参数数值一致不等于选择过程合规**；它们仍来自
test-ranked trial 2。

### 4.2 INTERPEP、PEPBIND、PEPNN 的实际模型不是 n_layers=3

更接近 checkpoint 生成时期的配置备份
[`config.py.before_biomamba_params_20260724:4-22`](./config.py.before_biomamba_params_20260724#L4-L22)
记录：

- interpep：`lr=0.000214, dropout=0.176, alpha=0.656, gamma=2.187`
- pepbind/pepnn：`lr=0.000150, dropout=0.300, alpha=0.700, gamma=2.000`
- `MODEL_CONFIG["n_layers"] = 2`

对三个 checkpoint 的 state-dict 键进行只读检查，均只包含 layer index
`0, 1`；以 `n_layers=2`、`d_model=128` 等配置实例化当前 `PepMamba` 后，
三个 checkpoint 均 strict load 成功。以 n_layers=3 则不会严格匹配。

因此，如果论文把这三个 checkpoint 作为最终主结果，实际设置应写
`n_layers=2`，不是任务中拟写的 `n_layers=3`。

当时 [`train.py`](./train.py) 的固定值为：

| 参数 | 实际代码值 | 证据 |
|---|---:|---|
| ce_weight | 0.5 | [`train.py:237-242`](./train.py#L237-L242) |
| weight_decay | 0.01 | [`train.py:231`](./train.py#L231) |
| batch_size | 16 | [`train.py:323-330`](./train.py#L323-L330) |
| scheduler_t0 | 15 | [`train.py:232-234`](./train.py#L232-L234) |
| class_weight_power | 0.5（平方根） | [`train.py:167-169`](./train.py#L167-L169) |
| max_pos_weight | 6.0 | [`train.py:156-169`](./train.py#L156-L169) |
| grad_clip | 1.0 | [`train.py:262`](./train.py#L262) |
| n_layers | 2 | checkpoint keys + 配置备份第 13 行 |
| epochs | CLI 默认 35 | [`train.py:300-305`](./train.py#L300-L305) |
| seed | CLI 默认 42 | [`train.py:300-310`](./train.py#L300-L310) |

限制：checkpoint metadata 未保存完整训练参数、总 epochs 或 seed，项目也无
对应日志，所以 epochs=35 和 seed=42 是“文件名/默认命令一致”的强证据，
但无法排除当时 CLI 覆盖。论文不应在缺少 run config 的情况下把它们表述为
已由 metadata 证明。

## 5. 每类结果的十项 test-selection 判断

| 结果类 | CKPT 仅 val | 超参数仅 val | 阈值仅 val | test 固定后评价 | test scan | scan 仅 debug | scan 进入 final | test 排 trial | 因 test 命名 final | 重复看 test |
|---|---|---|---|---|---|---|---|---|---|---|
| BITENET_FINAL_KEEP | 是 | **否** | **否** | 否 | 是 | 否 | 是 | 是 | 是 | 是 |
| 14 条论文 BITENET 消融 | 是 | **否（继承）** | 是 | 单条运行内是 | 否 | N/A | 否 | 汇总按 test 排 | 可能 | 是 |
| BioMamba 0.8574 override | 是 | **否（继承）** | 是 | 单次运行内是 | 否 | N/A | 否 | 无运行选择记录 | 高风险 | 是，至少五次 |
| BioMamba 同 seed 三次 | 是 | **否（继承）** | 是 | 单次运行内是 | 否 | N/A | 否 | 未保存 | 未证明 | 是 |
| PepBind extreme trial 11 | 是 | **否** | **否** | 否 | 是 | 否 | 是 | 是 | 是 | 是，20 trials |
| INTERPEP/PEPBIND/PEPNN checkpoint | 是 | 未知 | metadata 称 val | 未找到结果 | 未知 | 未知 | 未找到 | 未知 | 未知 | 未知 |
| 通用 test.py 输出 | N/A | N/A | **否，硬编码 0.65** | 否 | 是 | 代码称 debug | 未自动写文件 | N/A | N/A | 无日志 |

## 6. 是否可以不重新训练而规范重算

### 6.1 可以直接重算 test 指标

以下目录已有 validation/test labels/scores，无需 checkpoint、无需推理：

- 14 条 `PAPER_ABLATION_FINAL_4GROUPS` seed=42 源运行；
- 五条 `biomamba_rerun_original_env*`；
- 三条 `biomamba_original_same_seed_three_runs_20260723`；
- `BITENET_FINAL_KEEP/threshold_scan/`；
- PepBind extreme 的每个 trial；
- validation-targeted Mamba1/BioMamba 和 stacking 结果。

可用统一只读重算器执行：

1. 只在 validation labels/scores 上扫描 MCC；
2. 冻结阈值；
3. 在 test labels/scores 上计算 AUROC、AUPRC、MCC、F1、Precision、
   Sensitivity、Specificity；
4. 新输出建议放在
   `tuning_results/PAPER_CLEAN_REEVAL_20260724/<dataset>/<variant>/`。

本次按要求没有实际写入该目录。

### 6.2 三个 checkpoint 可不训练、但需要重新推理

INTERPEP、PEPBIND、PEPNN checkpoint 均可被当前模型以 `n_layers=2` 严格加载，
所以无需重新训练即可：

- 读取对应 `*_best_auroc_meta.pt` 的 validation threshold；
- 对 test split 做一次推理；
- 固定该阈值计算指标；
- 同时保存 labels/scores、完整 metadata 和 SHA256。

不能直接使用当前 `test.py`，因为其
[`128`](./test.py#L128) 硬编码 0.65。应使用新的独立 evaluator，输出到新目录，
不覆盖旧文件。

### 6.3 仅重算不能消除的污染

对 BITENET trial 2、14 条消融及 PepBind extreme，“重新按 validation 阈值算
一次 test”只能修复阈值字段，**不能修复超参数/trial 已由 test 选择**。

完全规范的最小方案：

1. 冻结一个只使用 train/validation 的超参数和架构选择规则；
2. 不读取 test 指标地选出唯一配置；
3. 对该配置重新训练（或使用事先已生成且从未按 test 筛选的 checkpoint）；
4. validation 选阈值；
5. test 只评价一次；
6. 最理想是使用新的外部测试集；因为现有 BITENET test 已在大量开发循环中
   被反复查看，单纯重新训练无法让它重新变成真正盲测集。

## 7. 最终分类

### A. 可以直接用于论文的结果

在本任务的严格 GREEN 定义下：**无**。

外部方法的原论文数字可以作为“文献报告结果”使用，但必须补齐原论文表号、
数据版本和引用；它们不是本项目复现结果。

### B. 可通过重新按 validation 阈值计算后使用

- 从数值重算角度：所有保存 validation/test predictions 的结果都可以。
- 从无泄漏论文结果角度：只有那些超参数和 trial 事先不依赖 test 的结果才可以。
  当前 14 条主消融不满足后一条件。
- INTERPEP/PEPBIND/PEPNN 可用现有 checkpoint 和 metadata threshold 重新推理，
  但因超参数 provenance 不完整，重测后最多先升为 YELLOW，不能自动变 GREEN。

### C. 必须重新按 validation 选择 trial 的结果

- BITENET full-search 20 trials；
- PepBind extreme 20 trials；
- BITENET single-model AUC860 refinement；
- 任何从五次 BioMamba seed=42 重跑中挑一条的主表；
- 由已经 test-visible 的大规模 base model 池构建的 stacking。

### D. 来源无法确认、暂时不能用于论文

- INTERPEP、PEPBIND、PEPNN 的最终 test 指标；
- 当前配置所指但实际不存在的
  `out_model/bitenet/biomamba_bitenet_seed42` 主结果；
- PepCA/PepNN/PepBind/InterPep 外部方法对比数值；
- 任何只存在于未提供的论文草稿、截图或聊天记录中的表。

### E. 最少需要重新运行的实验

1. **四数据集主模型**：冻结四套参数和模型结构后，建立统一 evaluator；
   三个已有 checkpoint 可先重测，BITENET 需先确定一个合规 checkpoint。
2. **BITENET 主结果**：至少重新以 validation-only 规则确定超参数/架构。
3. **四组消融**：若论文声称严格盲测，应用同一套 validation-only 公共参数重跑
   14 个 seed=42 变体；若报告均值，还应预先固定 seeds 后全部重跑。
4. **PepBind extreme**：废弃 test-ranked trial 11；用 validation 重新选唯一 trial。
5. **外部方法对比**：不一定需要运行，但必须补齐原始论文/官方实现来源。

### F. 需要重新训练还是只需重新测试

| 情况 | 最少动作 |
|---|---|
| INTERPEP/PEPBIND/PEPNN checkpoint | 只需规范重新测试；若要 GREEN，还需补齐超参数来源 |
| BITENET_FINAL_KEEP 的 0.605 | 可先用现有预测重算 0.63 的 test，但超参数仍污染 |
| BITENET test-selected trial 2 | 为获得干净 provenance，需要 validation-only 选配置并重新训练/固定 checkpoint |
| 14 条 BITENET 消融 | 严格方案需要用干净公共参数重新训练 |
| PepBind trial 11 | 必须 validation-only 重选；严格方案需重训所选配置并一次性评价 |
| 外部方法文献数字 | 不需训练；需来源引用 |

### G. 论文方法部分可以安全写出的规则

只有在后续规范重跑实际遵守后，才可以安全写：

> At each epoch, model checkpoints were selected solely according to validation
> AUROC, with validation AUPRC used as a deterministic tie-breaker. The decision
> threshold was selected exclusively on the validation set by maximizing MCC
> over the pre-specified threshold grid. After the hyperparameters, architecture,
> checkpoint, and threshold had been frozen, the test set was evaluated once;
> test labels were not used for threshold tuning, hyperparameter selection,
> trial ranking, or model selection.

当前历史结果只能安全写到：

> Within each saved ablation run, the checkpoint was selected by validation
> AUROC (validation AUPRC tie-break), and the recorded per-run threshold was
> selected by validation MCC before being applied to test.

不能安全写“所有超参数只由 validation 选择”或“test 只使用一次”，因为
BITENET/PepBind 历史脚本和 final 目录提供了相反的直接证据。

