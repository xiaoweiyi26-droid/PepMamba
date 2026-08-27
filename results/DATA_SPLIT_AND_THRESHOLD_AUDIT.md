# PepMamba 四数据集数据划分与阈值审计

审计对象：`.`
数据集：`bitenet`、`interpep`、`pepbind`、`pepnn`
审计日期：2026-07-24
审计方式：只读检查项目代码、CSV、日志、归档、文件时间、远端公开仓库和原论文；未修改代码、数据或既有实验结果。

## A. 数据处理流程总览

### A.1 当前能够复原的实际流程

```text
四个既有公开 benchmark 的原始 training/test 划分
    ↓
公开 PepCA 仓库提供的 *_train.csv、*_val.csv、*_test.csv
    ↓
PepMambaDataset 按 “{dataset}_{mode}.csv” 直接读取
    ↓
按完整蛋白质–肽复合物行加载 Prot_seq、Pep_seq 和整条蛋白质残基标签
    ↓
从 protein_esm2.pkl / peptide_esm2.pkl 按序列取 ESM-2 特征
    ↓
train 用于参数更新，val 用于 checkpoint/阈值选择，test 用于最终评价
```

直接读取证据：

- [`dataloaders.py`](./dataloaders.py#L19-L30) 第 23–24 行拼接并读取
  `"{data_set}_{mode}.csv"`；第 48–56 行从同一行读取 `Prot_seq`、`Pep_seq`、
  `label` 和相应序列特征。
- [`dataloaders.py`](./dataloaders.py#L63-L75) 第 63–65 行仅在特征长度与标签长度
  不一致时同步截断；第 67–73 行只对训练特征做噪声/通道遮蔽，不重新划分数据。
- [`train.py`](./train.py#L214-L229) 第 217–228 行直接构建现成的 `train` 和 `val`
  dataset/loader；没有调用任何 split 函数。
- [`test.py`](./test.py#L115-L122) 第 115–122 行直接读取现成的 `test` CSV。
- [`train.py`](./train.py#L323-L330) 第 326–329 行给出当前实际数据目录和
  `protein_esm2.pkl`、`peptide_esm2.pkl` 路径。

### A.2 原始 benchmark 与本地别名

PepCA 原论文把四套 benchmark 记为 TS092、TS251、TS639、TS125。当前文件的
test 样本数和残基统计与论文表格逐项完全一致，因此对应关系为：

| 本地别名 | 原论文名称 | 原始来源 | 当前 test 行数 |
|---|---|---|---:|
| `pepnn` | TS092 | PepNN | 92 |
| `interpep` | TS251 | InterPep | 251 |
| `pepbind` | TS639 | PepBind | 639 |
| `bitenet` | TS125 | SPRINT-Seq/PepPI benchmark | 125 |

外部依据：

- [PepCA 原论文的数据集说明和 Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC11465048/)
  将 TS092、TS251、TS639、TS125 分别归于 PepNN、InterPep、PepBind、SPRINT-Seq，
  并给出与本地数据完全一致的 training/test 数量。
- [公开 PepCA GitHub 数据目录](https://github.com/cloudaner115/PepCA/tree/main/data)
  直接发布了与本地同名的 12 个 train/val/test CSV，但仓库根目录没有提供
  preprocess/split 脚本。

注意：项目没有解释为什么 TS125 在代码中命名为 `bitenet`。因此论文中宜写
“TS125（项目内别名 `bitenet`）”，不要把本地文件名直接当作原 benchmark 的正式名称。

### A.3 找到和未找到的处理文件

已找到：

- 通用 CSV/PKL loader：[`dataloaders.py`](./dataloaders.py#L11-L89)。
- 训练/验证读取：[`train.py`](./train.py#L214-L229)。
- 测试读取：[`test.py`](./test.py#L115-L122)。
- 调参读取：[`tune.py`](./tune.py#L67-L83)。
- BITENET 论文消融专用读取：
  [`run_bitenet_pretrained_repr_ablation.py`](./run_bitenet_pretrained_repr_ablation.py#L540-L564)。
- ESM 特征生成：[`esm.py`](./esm.py#L20-L29) 加载并冻结 ESM-2，
  第 40–47 行清洗非标准字符并对序列去重，第 69–79 行生成表示并写入 PKL。
- README 只说明训练前生成 ESM 特征和四个可选数据集：
  [`README.md`](./README.md#L18-L32)。

未找到：

- 原始 PDB/BioLiP 数据下载和解析脚本。
- 接触距离生成残基标签的脚本。
- 生成四套 `*_train.csv` / `*_test.csv` 的脚本。
- 从原始 training set 生成 `*_val.csv` 的脚本。
- `train_test_split`、GroupShuffleSplit、按蛋白质 ID 分组划分或按序列 cluster
  划分这 12 个当前 CSV 的代码。
- CD-HIT、BLASTCLUST、MMseqs/MMseqs2 的命令、shell 脚本或运行日志。

项目归档 `tihuan.zip` 只含代码文件，不含数据生成脚本；其文件清单中只有
`config.py`、`dataloaders.py`、`esm.py`、模型/训练/测试等 13 个文件。

## B. 四个数据集逐一分析

### B.1 BITENET（TS125）

原始来源：

- 本地 `bitenet_test.csv` 的 125 行、30,870 个残基、1,716 个正残基和
  29,154 个负残基与 PepCA 论文 TS125 test 完全一致。
- 当前 `bitenet_train.csv + bitenet_val.csv` 合计 1,115 行、266,434 个残基、
  14,829 个正残基和 251,605 个负残基，与论文 TS125 training 完全一致。
- 因此当前 val 位于原始 training 侧，而不是从原始 test 中划出；依据是上述四项
  计数恒等式以及 train/val 与 test 的配对零重叠。

validation：

- 实际行数为 train 1,040、val 75，即当前实现比例为 93.2735% / 6.7265%。
- **当前代码只能确认读取了独立的 val 文件，无法从现有文件确定其生成方式。**
- 找不到 split seed；训练默认 seed=42（[`train.py`](./train.py#L21-L29)、
  第 300–310 行）不能当作数据划分 seed。
- train–val 存在 15 条完全相同的蛋白质序列，但没有完全相同蛋白质–肽对，
  所以该划分明确不是严格的 protein-disjoint split，也没有在后续 train/val
  间维持 30% 蛋白质一致性上限。
- 15 个重复蛋白对应的 PDB/肽对可由
  `data/bitenet_train.csv` 与 `data/bitenet_val.csv` 的 `Prot_seq` 等值连接复核；
  其中包括 `3c3o_A_B ↔ 3C3R_A_B`、`3pob_A_B ↔ 3pob_AB`、
  `2k17_A_P ↔ 2k17_A_P`、`3KV4_A_B ↔ 3kv4_A_B` 等。

最终训练实际读取：

- 通用训练入口读取
  [`data/bitenet_train.csv`](./data/bitenet_train.csv) 和
  [`data/bitenet_val.csv`](./data/bitenet_val.csv)；
  通用测试入口读取 [`data/bitenet_test.csv`](./data/bitenet_test.csv)。
- BITENET 消融 runner 又在
  [`run_bitenet_pretrained_repr_ablation.py`](./run_bitenet_pretrained_repr_ablation.py#L540-L564)
  第 552–564 行显式记录和读取同三份文件。

### B.2 INTERPEP（TS251）

原始来源与清洗：

- 当前 train+val 为 225+26=251 行；正/负残基为
  5,957+670=6,627 和 43,255+4,606=47,861，精确还原 PepCA 论文 TS251 training。
- 当前 test 为 251 行、6,617 个正残基、51,987 个负残基，精确匹配 TS251 test。
- InterPep 原始数据来自 2016-05-19 的 PDB：较大蛋白链与 5–25 aa 肽链形成复合物，
  NMR 或分辨率不差于 3 Å 的晶体结构；还过滤共享界面小于 400 Å²和肽埋藏超过
  60% 的样本。[InterPep Methods, Dataset](https://www.nature.com/articles/s41598-019-38498-7#Sec2)
- InterPep 原研究使用 **BLASTCLUST** 在较长蛋白链上按 30% 序列一致性聚类；
  不是 CD-HIT。[InterPep 论文第 98–104 行所示 Dataset 方法](https://www.nature.com/articles/s41598-019-38498-7#Sec2)

validation：

- 当前比例为 train 89.6414%、val 10.3586%（225:26）。
- train/val/test 间完全相同蛋白质、完全相同蛋白质–肽对和相同 `pdb_id` 均为 0。
- 零“完全相同序列”重叠不能证明小于 30% identity，也不能区分按行随机划分、
  按蛋白质划分或聚类划分。
- **当前代码只能确认读取了独立的 val 文件，无法从现有文件确定其生成方式。**
  split seed 和是否分层均未知。

最终文件：

- [`data/interpep_train.csv`](./data/interpep_train.csv)
- [`data/interpep_val.csv`](./data/interpep_val.csv)
- [`data/interpep_test.csv`](./data/interpep_test.csv)

### B.3 PEPBIND（TS639）

原始来源：

- 当前 train+val 为 576+64=640 行；正/负残基为
  7,345+914=8,259 和 133,643+15,323=148,966，与 PepCA 论文 TS639 training
  精确一致。
- 当前 test 为 639 行、8,490 个正残基、141,838 个负残基，与 TS639 test 精确一致。
- PepBind benchmark 的 1,279 个样本原始上被随机等分为 640 training 和 639 test；
  该来源关系也见 [PepENS 对 PepBind Dataset 2 的回溯说明](https://pmc.ncbi.nlm.nih.gov/articles/PMC12541375/)。

validation：

- 当前 train/val 恰为 90%/10%（576:64）。
- 三分集间完全相同蛋白质、完全相同蛋白质–肽对和相同 `pdb_id` 均为 0。
- **当前代码只能确认读取了独立的 val 文件，无法从现有文件确定其生成方式。**
  虽然数量恰为 90:10，但不能据数量自行断言使用了 sklearn 随机划分，也不能断言
  seed=42。

最终文件：

- [`data/pepbind_train.csv`](./data/pepbind_train.csv)
- [`data/pepbind_val.csv`](./data/pepbind_val.csv)
- [`data/pepbind_test.csv`](./data/pepbind_test.csv)

### B.4 PEPNN（TS092）

原始来源：

- 当前 train+val 为 2,517+311=2,828 行；正/负残基为
  76,604+8,929=85,533 和 535,373+68,414=603,787，与 PepCA 论文 TS092 training
  精确一致。
- 当前 test 为 92 行、2,523 个正残基、20,650 个负残基，与 TS092 test 精确一致。
- PepNN 原论文说明：删除了与 training 或 validation 中序列在 70% coverage 下
  超过 30% identity 的 test 序列。
  依据：[PepNN 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9135736/)。

validation：

- 当前比例为 train 89.0028%、val 10.9972%（2,517:311）。
- 三分集间完全相同蛋白质、完全相同蛋白质–肽对和相同 `pdb_id` 均为 0。
- train 内 2,517 行对应 1,846 条唯一蛋白，val 内 311 行对应 220 条唯一蛋白，
  说明划分单位不是残基；每行仍是一个完整蛋白质–肽对。
- **当前代码只能确认读取了独立的 val 文件，无法从现有文件确定其生成方式。**
  split seed、是否按蛋白分组、是否沿用 PepNN 原始 train/validation 均无法仅由当前文件确定。

最终文件：

- [`data/pepnn_train.csv`](./data/pepnn_train.csv)
- [`data/pepnn_val.csv`](./data/pepnn_val.csv)
- [`data/pepnn_test.csv`](./data/pepnn_test.csv)

## C. train/val/test 统计表

统计口径：

- 样本 = CSV 数据行 = 一个蛋白质–肽对。
- 唯一蛋白/肽/配对按 `Prot_seq` / `Pep_seq` / 二者元组精确字符串去重。
- 正负残基直接统计 `label` 中的 `1` / `0`。
- 已检查 12 个 CSV，所有行均满足 `len(label) == len(Prot_seq)`。

| 数据集 | split | 样本数 | 唯一蛋白 | 唯一肽 | 唯一配对 | 正残基 | 负残基 |
|---|---|---:|---:|---:|---:|---:|---:|
| bitenet | train | 1,040 | 1,040 | 940 | 1,040 | 13,857 | 236,527 |
| bitenet | val | 75 | 75 | 75 | 75 | 972 | 15,078 |
| bitenet | test | 125 | 125 | 123 | 125 | 1,716 | 29,154 |
| interpep | train | 225 | 225 | 222 | 225 | 5,957 | 43,255 |
| interpep | val | 26 | 26 | 26 | 26 | 670 | 4,606 |
| interpep | test | 251 | 251 | 248 | 251 | 6,617 | 51,987 |
| pepbind | train | 576 | 576 | 529 | 576 | 7,345 | 133,643 |
| pepbind | val | 64 | 64 | 64 | 64 | 914 | 15,323 |
| pepbind | test | 639 | 639 | 591 | 639 | 8,490 | 141,838 |
| pepnn | train | 2,517 | 1,846 | 2,516 | 2,517 | 76,604 | 535,373 |
| pepnn | val | 311 | 220 | 311 | 311 | 8,929 | 68,414 |
| pepnn | test | 92 | 90 | 92 | 92 | 2,523 | 20,650 |

数据依据为 [`data/`](./data) 下对应 CSV。所有 12 个文件的时间戳均为
2026-03-22 01:37:52–01:37:52（+0800）；相同时间戳只说明它们很可能被同批复制/解包，
不能证明 split 算法或 seed。

## D. 数据重叠与潜在泄漏检查

### D.1 完全相同蛋白质重叠

| 数据集 | train∩val | train∩test | val∩test | 三者共同 |
|---|---:|---:|---:|---:|
| bitenet | **15** | 0 | 0 | 0 |
| interpep | 0 | 0 | 0 | 0 |
| pepbind | 0 | 0 | 0 | 0 |
| pepnn | 0 | 0 | 0 | 0 |

### D.2 完全相同蛋白质–肽对重叠

| 数据集 | train∩val | train∩test | val∩test | 三者共同 |
|---|---:|---:|---:|---:|
| bitenet | 0 | 0 | 0 | 0 |
| interpep | 0 | 0 | 0 | 0 |
| pepbind | 0 | 0 | 0 | 0 |
| pepnn | 0 | 0 | 0 | 0 |

结论：

1. 没有完全相同的蛋白质–肽对跨集合重复。
2. BITENET 存在明确的 protein-level train–val leakage/非独立性：相同蛋白、
   不同肽对跨 train 和 val。若研究问题强调“未见蛋白质泛化”，该 val 不能被描述为
   protein-disjoint。
3. 其余三套数据没有**完全相同**蛋白重叠，但本次没有发现/运行产生这些 split 的
   30% 聚类流程，因此不能进一步声称不存在 30%–100% 的近同源序列。
4. 划分不是残基层面的：loader 按完整 CSV 行返回整条蛋白标签
   （[`dataloaders.py`](./dataloaders.py#L48-L75)）。
5. 对 BITENET，可排除“严格按蛋白质划分”；对另外三套，现有证据无法区分
   按样本随机、按蛋白质还是按 cluster 划分。

### D.3 30% 序列一致性控制的适用范围

- PepCA 原论文只说明每个上游 benchmark 经 BLASTCLUST 或 MMseq2 等工具独立处理，
  将序列一致性降低到 30%；这支持原始 benchmark/training-test 构造层面的控制。
- 当前项目没有 CD-HIT 调用；没有 BLASTCLUST/MMseqs2 调用、参数文件或 cluster ID。
- BITENET train–val 有 100% 相同蛋白序列，直接证明原论文的 30% 控制**没有同时约束
  当前后续 train/validation 边界**。
- 对 interpep/pepbind/pepnn，零完全相同序列重叠不足以证明 30% 控制延伸到 train/val；
  必须补充原 split 脚本或重新运行序列聚类/全对全一致性审计后才能作此表述。
- 因此，对问题“论文 30% 控制只针对原始 train/test，还是也用于后续 train/val？”
  当前安全答案是：**能够确认的是上游 benchmark 控制；无法确认后续 val 也受控制，
  且 BITENET 的实测结果明确否定了这种控制。**

## E. 阈值选择流程

项目中存在多套流程，必须区分。

### E.1 当前通用 `train.py` / `test.py`

checkpoint：

- 每个 epoch 在 val 上计算 AUROC/AUPRC 和最佳阈值
  （[`train.py`](./train.py#L174-L210)）。
- checkpoint 只按 **validation AUROC** 提升保存；AUPRC 被记录但不作 tie-break
  （[`train.py`](./train.py#L244-L293)，尤其第 280–293 行）。

阈值：

- `train.py` 在 val 标签上最大化 MCC
  （[`train.py`](./train.py#L61-L87)）。
- 候选为 `np.arange(0.10, 0.90, 0.01)`，即实际 0.10–0.89、步长 0.01。
- val 最佳阈值写入 `*_best_auroc_meta.pt`
  （[`train.py`](./train.py#L283-L293)）。

**主测试脚本存在实现错误/不一致：**

- [`test.py`](./test.py#L126-L135) 在 meta 存在时没有读取
  `meta["best_threshold"]`，而是硬编码 `threshold = 0.65`；随后把来源文字覆盖成
  `"validation-selected threshold"`。因此当前通用 `test.py` 的报告阈值实际上不是
  从 meta 读取的 validation 阈值。
- [`test.py`](./test.py#L136-L152) 还使用 test 标签重新搜索一次最大 MCC 阈值，
  名为 `oracle_thresh`，并标记为 debug-only。该 oracle 没有用于第 135 行的正式
  metrics，但它证明项目中确实存在 test-label threshold search 代码。

### E.2 当前 BITENET 论文消融 runner

该流程较规范：

- checkpoint 按 `(validation AUROC, validation AUPRC)` 字典序选择
  （[`run_bitenet_pretrained_repr_ablation.py`](./run_bitenet_pretrained_repr_ablation.py#L606-L624)）。
- 最佳 checkpoint 载入后，仅在 validation predictions 上扫描阈值
  （同文件第 626–634 行）。
- 阈值范围 0.10–0.90（含 0.90）、步长 0.01；主指标 MCC，平局依次看 precision、
  sensitivity（[`run_bitenet_pretrained_repr_ablation.py`](./run_bitenet_pretrained_repr_ablation.py#L280-L286)）。
- test 直接使用 validation 选中的固定阈值
  （同文件第 636–642 行），结果元数据也声明
  `selected_threshold_source = "validation_mcc_scan"`（第 660–662 行）。

### E.3 存在 test 标签重搜/挑选的历史代码

以下结果不能笼统描述为“阈值只由 validation 决定”：

- [`tune_pepbind_extreme.py`](./tune_pepbind_extreme.py#L450-L507)
  第 464–467 行同时扫描 val 和 test，`result["test"]` 保存 test 最优阈值；
  第 495–497 行才另存 val 阈值作用于 test 的结果。
- 同文件第 525–573 行根据 test ranking 生成推荐配置，并把
  `best["test"]["threshold"]` 写入 `PEPBIND_THRESHOLD`；报告第 573 行也明确承认
  最终 trial ranking 使用 test scan。
- [`tuning_results/BITENET_FINAL_KEEP/scripts/refine_bitenet_single_auc860.py`](./tuning_results/BITENET_FINAL_KEEP/scripts/refine_bitenet_single_auc860.py#L293-L304)
  第 298–304 行直接对 `test_labels/test_scores` 扫描并选择阈值。
- `BITENET_FINAL_KEEP` 中的主 extreme runner 虽然把 validation 阈值固定到 test，
  但多 trial 均计算 test 指标；如果再按这些 test 指标挑 trial，也构成测试集反复查看，
  应与最终一次性 test evaluation 区分。

## F. 能够确定的事实

1. 当前训练不动态划分数据，而是直接读取 12 个现成 CSV。
2. 每个 CSV 行是完整的蛋白质–肽对；监督标签是蛋白质全长残基二分类串。
3. 四套当前 train+val 的行数和正/负残基数均精确等于 PepCA 论文原始 training
   统计；四套当前 test 均精确等于论文 test 统计。
4. 因而 val 来自原始 training 侧，而非原始 test 侧；但生成程序已缺失。
5. 当前实现的 train/val 实际比例分别为：
   - bitenet 93.2735% / 6.7265%
   - interpep 89.6414% / 10.3586%
   - pepbind 90% / 10%
   - pepnn 89.0028% / 10.9972%
6. 没有任何跨 split 的完全相同蛋白质–肽对。
7. BITENET train/val 有 15 条完全相同蛋白质；其余三套精确蛋白重叠为 0。
8. 项目内没有 CD-HIT、BLASTCLUST、MMseqs2 的实际调用记录。
9. 本地项目目录不是 Git working tree；`git rev-parse` 和针对该目录的 `git log`
   均返回 “not a git repository”。因此无法利用本地 Git 历史追溯 val 的生成 commit。
10. 公开 GitHub 仓库同样直接发布 split CSV，根目录未提供 validation 生成脚本。
11. 主 `train.py` 按 validation AUROC 选 checkpoint、按 validation MCC 选阈值；
    主 `test.py` 却硬编码 0.65，未实际读取保存的 validation 阈值。
12. BITENET 论文消融 runner 正确地只在 validation 上选阈值并固定应用于 test。

## G. 当前无法确定的问题

1. 四套 `*_val.csv` 的确切生成脚本和命令。
2. split seed；不能把训练 seed=42 自动等同于数据 split seed。
3. 预期 train/val 比例。表 C 只能报告实际实现比例，不能反推设计比例。
4. 是否使用随机抽样、分层抽样、按 PDB、按蛋白质或按 cluster 划分。
5. interpep/pepbind/pepnn 的 train–val 是否在 30% identity 下真正隔离。
6. 上游各 benchmark 的所有原始清洗细节是否在当前 CSV 中被完全保留；本地没有原始
   PDB/BioLiP 输入和转换脚本可逐行复算。
7. TS125 为什么在代码中命名为 `bitenet`。
8. 公开仓库的完整数据 commit 历史在当前本地副本中不存在；只能确认公开仓库当前
   tree 含这些文件，不能由本地文件确定最初生成者和命令。

## H. 论文 Materials and Methods 中可以安全采用的表述

建议采用以下保守表述：

> We used the four benchmark datasets distributed with the public PepCA
> implementation: TS092 (PepNN; local alias `pepnn`), TS251 (InterPep),
> TS639 (PepBind), and TS125 (local alias `bitenet`). The published test
> sets were retained unchanged. Model development used the repository-provided
> training and validation CSV files. Their realized train/validation sample
> counts were 2517/311, 225/26, 576/64, and 1040/75, respectively. Each sample
> represented one protein–peptide pair, and residue-level binary labels covered
> the complete receptor protein sequence.

> Upstream benchmark construction applied sequence-redundancy control near
> 30% identity using benchmark-specific tools such as BLASTCLUST or MMseqs2.
> The current repository does not include the script or random seed used to
> derive its validation CSV files; therefore, we do not claim that the later
> train/validation split was cluster-disjoint. Exact-sequence auditing found
> no duplicated protein–peptide pairs across splits. Fifteen receptor protein
> sequences were shared between the TS125 local training and validation files.

若论文结果使用 BITENET 消融 runner，可安全补充：

> The checkpoint was selected by validation AUROC, with validation AUPRC as
> the tie-breaker. The decision threshold was selected on the validation set
> by maximizing MCC over thresholds from 0.10 to 0.90 in increments of 0.01,
> and was then applied unchanged to the test set.

若论文结果来自通用 `train.py/test.py`，在修正 `test.py` 前不要写最后一句，因为当前
测试脚本实际使用硬编码 0.65。

## I. 仍需补充或修正的内容

数据可复现性：

1. 找回或重建四套 validation 生成脚本，并保存：
   输入文件哈希、划分单位、比例、seed、是否 stratify/group、输出 SHA-256。
2. 若目标是未见蛋白泛化，必须重新处理 BITENET 的 15 个 train–val 同蛋白案例，
   或明确把当前 val 描述为 pair-level development split。
3. 对四套 train/val 运行统一的 MMseqs2/CD-HIT/BLAST identity audit；仅检查完全相同
   序列不能验证 30% 阈值。
4. 保存上游原始 training/test 文件或下载 URL、版本、日期与许可信息。
5. 明确 `bitenet` 与 TS125/SPRINT-Seq 的命名关系。

阈值与评估：

1. 修正 [`test.py`](./test.py#L126-L135)，应读取 `meta["best_threshold"]`，而不是
   硬编码 0.65；同时修正误导性的 `threshold_source`。
2. 最终报告路径中移除或隔离 test oracle threshold；debug 扫描也应避免进入正式流程。
3. PEPBIND 不应使用
   [`tune_pepbind_extreme.py`](./tune_pepbind_extreme.py#L464-L573)
   的 test-selected threshold/config 作为无偏最终结果；应以 validation 选择参数和阈值，
   test 只运行一次。
4. BITENET `refine_bitenet_single_auc860.py` 的 test threshold scan 只能标记为事后分析，
   不能作为最终方法学阈值。
5. 在最终论文中为每个结果表明确标注所用 runner、checkpoint 规则、阈值来源和是否曾
   参与多 trial test ranking，避免把不同流程混为一谈。

