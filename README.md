# 学习完成预测审阅

容器环境固定在仓库 Dockerfile 的 Python 3.11.14 镜像摘要，数值分析依赖与传递依赖在 requirements.txt 中精确锁定。执行 `docker build -t local-analytics .` 后，`docker run --rm --network none local-analytics` 运行内置例子；`docker run --rm --network none local-analytics python -m unittest discover -s tests -v` 运行测试。构建时安装依赖，运行时无需联网；非容器环境使用 Python 3.11，并先执行 `python -m pip install -r requirements.txt`。

该原型用合成课程记录说明预测时点、特征与复盘结果之间的关系，提供固定版本的数值和建模依赖。

运行 `python -m unittest discover -s tests -v`；`python -m learning.demo` 会输出一个批次的预测审阅摘要。模型是用于验证流程的简单基线，不应直接用于学员处分或资源分配。

`learning.advising.AdvisingDesk` 连接风险分数与辅导席位，按预测完成概率从低到高提出名单，分数并列时按学员编号排序。席位按 `cohort` 配置，确认后的 `session_id` 不能换成另一份名单。活动的发生时间和入账时间分别为 `occurred_at`、`recorded_at`；当前模型参数由调用方提供，没有训练或模型优选流程。所有时间为 UTC 秒。
