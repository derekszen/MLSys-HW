# Homework 1 / Homework 2 提交说明

**姓名：Zhuoxi Zeng　学号：2501213387。两次作业技术内容已完成，报告与附件分别准备；尚未上传教学网。**

**作业二成员信息：** 已补充 Steven Walxim（2501112130）的离线复核工具、测试可移植性修复及已有证据复核贡献；这不代表新增 NPU 实测。成员 3 的姓名、学号与实际贡献仍为“待确认”。提交前核对成员信息并重新生成 HW2 PDF/ZIP；目前不能将该占位版本当作最终提交稿。作业一保持不变。

## 需要上传的文件

所有生成文件位于项目根目录的 `submissions/`：

| 作业入口 | 主报告（优先上传） | 代码与实验附件（入口允许时一并上传） |
| --- | --- | --- |
| 第1次作业 / 计算子系统 | `2501213387_Zhuoxi_Zeng_HW1_Report.pdf` | `2501213387_Zhuoxi_Zeng_HW1_Submission.zip` |
| 第2次作业 / 通信子系统 | `2501213387_Zhuoxi_Zeng_HW2_Report.pdf` | `2501213387_Zhuoxi_Zeng_HW2_Submission.zip` |

附件包含报告 PDF/Markdown/HTML、原题、完整对应阶段的 TokenSpeed 源码、固定版本 patch、测试与复现脚本、原始日志/结果和压缩 trace，以及文件 SHA256 清单。**HW1 附件使用冻结的 HW1 源码；HW2 附件使用 HW1 + HW2 增量源码。** HW2 包中的 HW1 材料是前置依据，不应拿该包的源码重做 HW1 基线。

不包含模型权重、虚拟环境、Git 私有元数据、校园密码、二次认证信息或个人 VPN 配置。已有压缩 trace/array 不再附重复的未压缩副本。历史失败有明确标识，最新通过配置见各报告，不把失败结果改写为成功。

## 提交前只需完成以下步骤

1. **审核两份 PDF。** 核对姓名、学号、中文文字和表格；重点看结论、配置与贡献声明。开头记录需求/实验目标确认、资源访问协调和回归/修复/交付要求，若有其他成员须据实补齐。按本人选择，AI 辅助说明由本人填写在网站的提交评论中，不重复写入报告或附件说明。本条为本地操作提醒，不复制进提交 ZIP。
2. **确认课程规则。** 原题没有指定 PDF/ZIP 格式、附件大小、上传数量或教学网具体网址；以实际作业入口及教师最新通知为准，并确认 AI 辅助政策。不能保证满分，评分由教师决定。
3. **分别打开教学网的第1次、第2次作业入口。** 对应上传各自 PDF；若支持附加材料，再附相应 ZIP。若只允许一个文件且支持 ZIP，ZIP 已含 PDF；若只收 PDF，先按入口要求交 PDF，其他材料按教师允许的方式补充。不要把两次作业交错。
4. **完成正式提交并核验回执。** 点击平台的“提交”而不只保存草稿；确认附件名、提交状态/时间，并保留回执或截图。仓库中的“已准备”不等于平台“已提交”。

## 本地核验与重新生成

```bash
# 从项目根目录执行；不需要登录课程服务器。
(cd submissions && sha256sum -c SHA256SUMS)
python scripts/verify_submission.py submissions/2501213387_Zhuoxi_Zeng_HW1_Submission.zip
python scripts/verify_submission.py submissions/2501213387_Zhuoxi_Zeng_HW2_Submission.zip

# 修改 report.md / 贡献声明后，重新生成 PDF、ZIP 和清单：
python scripts/build_submissions.py
```

生成器使用本地 Pandoc、无界面的 Chromium 和中文字体；只在临时目录重建对应源码，不改变现有 TokenSpeed 工作树或 frozen patch，不访问教学网。平台的文件类型/大小限制尚未核对；如限制与材料不符，需要按入口规则调整附件而不是删除主要实验结论。

技术入口：[HW1 报告](homework-1/report.md)、[HW1 要求核对](homework-1/REQUIREMENTS.md)、[HW2 报告](homework-2/report.md)、[HW2 要求核对](homework-2/REQUIREMENTS.md)。
