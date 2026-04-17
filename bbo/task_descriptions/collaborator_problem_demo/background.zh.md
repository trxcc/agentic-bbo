# 背景

设想一个协作者希望 benchmark 一个 agent，用来调节实验室成像流程中的数据处理 pipeline。
每次 evaluation 都运行一个 surrogate simulator，用来近似图像质量和吞吐率。
benchmark 设计者希望任务包既便于人读，也便于程序校验，还能直接被 LLM agent 消化。

这个目录不是最小的 smoke test，而是一个偏文档驱动的示例，展示如何描述一个更真实的问题。
