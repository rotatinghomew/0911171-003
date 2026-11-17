# 学习完成预测审阅

容器环境固定在仓库 Dockerfile 的 Python 3.11.14 镜像摘要，数值分析依赖与传递依赖在 requirements.txt 中精确锁定。执行 `docker build -t local-analytics .` 后，`docker run --rm --network none local-analytics` 运行内置例子；`docker run --rm --network none local-analytics python -m unittest discover -s tests -v` 运行测试。构建时安装依赖，运行时无需联网；非容器环境使用 Python 3.11，并先执行 `python -m pip install -r requirements.txt`。

该原型用合成课程记录说明预测时点、特征与复盘结果之间的关系，提供固定版本的数值和建模依赖。

运行 `python -m unittest discover -s tests -v`；`python -m learning.demo` 会输出一个批次的预测审阅摘要。模型用于验证流程，不应直接用于学员处分或资源分配。

## 席位流程

`learning.advising.AdvisingDesk` 连接风险分数与辅导席位，按预测完成概率从低到高提出名单，分数并列时按学员编号排序。席位按 `cohort` 配置，确认后的 `session_id` 不能换成另一份名单。活动的发生时间和入账时间分别为 `occurred_at`、`recorded_at`；所有时间为 UTC 秒。`propose` 接受已训练模型（`model=`）或手工参数（`parameters=`），前者会在名单中记录 `model_version` 以便审计。

## 标签口径（learning.labels）

退课、尚未结课、结课后修正不一律记为失败：`completed` 记 1、`failed` 记 0；`withdrawn`（退课）单独排除；尚未结课或结课后没有结果记录一律视为未知并排除；结课后修正以入账时间为准，只影响修正入账之后的标签时点。

## 回放评估（learning.evaluation）

`python -m learning.pipeline evaluate --data synthetic --out report.json` 按课程批次回放历史：第 k 批在其预测时点（开课 + `lead_days`）训练，只用更早批次且结果在训练时点前入账的记录，再预测第 k 批；同一批次的记录绝不随机拆开。报告同时给出：

- 区分度（AUC、平均精度）与校准（Brier、log loss、分箱校准表），附 bootstrap 95% 区间；
- 各班既定席位下会选中的学员名单；
- 各群体（班次及报名属性）的失败学员捕获率与漏选信号，样本不足的群体只标注、不下结论；
- 可解释基线（逻辑回归，系数可读）与候选方法（梯度提升树）的逐批对比。

`--data` 也可以是包含 `enrollments.json`、`activity.json`、`outcomes.json`（可选 `seats.json`、`meta.json`）的目录；`synthetic` 为确定性的内置合成历史，仅用于本地验证。

## 训练到名单的可重现路径

```
python -m learning.pipeline evaluate --data synthetic --out report.json   # 候选比较，只读
python -m learning.pipeline train --data synthetic --artifacts artifacts  # 写入候选版本
python -m learning.pipeline promote --artifacts artifacts --version <v>   # 人工晋升正式模型
python -m learning.pipeline propose --data synthetic --artifacts artifacts \
    --cohort batch-6 --session week1 --reservations reservations          # 生成并确认名单
```

候选与正式分离：`train` 只写 `artifacts/candidates/`（内容寻址，同数据同版本号，重复训练是幂等的）；正式流程只读 `artifacts/champion.json` 指向的版本，`promote` 是显式人工动作，工具不自动晋升。重新训练不会改动正式指针，也不会改动 `reservations/` 中已确认的辅导名单；同号不同名单会被拒绝。选定的模型连同特征口径（`learning.features.FEATURE_NAMES`）直接被席位流程使用，训练与Serving共用同一套时点特征。

## 边界

预测风险不等于辅导效果。本仓库只做预测与名单管理，不提供任何因果结论；是否上线、是否晋升由人依据回放报告（含样本支持与漏选情况）决定，不能仅凭一个平均分数。模型产物以 joblib 序列化，只应加载本流程自己生成的 artifacts。
