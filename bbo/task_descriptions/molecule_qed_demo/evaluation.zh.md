# 评估协议

- 数据来源：缓存后的 `examples/Molecule/zinc.txt.gz`；evaluator 会读取其中的 `zinc.txt` 成员文件，并把 249455 条 SMILES 暴露成一个类别池。
- 对每次 evaluation，都先用 `rdkit.Chem.MolFromSmiles` 解析候选字符串，再计算 `rdkit.Chem.QED.qed`，并返回 `qed_loss = 1.0 - qed`。
- 对于无效分子，`qed = 0.0`，因此其主目标就是最差可能的 `qed_loss = 1.0`。
- evaluator 在任务侧是确定性的，而且与 seed 无关。主目标记为 `qed_loss`；同时记录原始 `qed`，并在 metadata 中写入被选中的 `smiles` 以及 `valid_smiles`。
- 若主目标完全相同，则按更早写入日志的 trial 处理，因为 benchmark 会以 append-only 顺序 replay 运行记录。
