# CLAUDE.md

## 第一性原则
在任意时刻，确保你遵循以下原则：

1. 在你对本项目进行分析以及开发时，除非用户指明使用别的语言，否则全部基于中文进行分析与回答。如果你需要了解本项目的运行环境及其他原理，可以参考@README.md
2. **在我明确提出要求之前，不要开始撰写代码；** 我需要确保你在正确的逻辑上工作；
3. **只撰写我明确要求的代码，不要延伸编写别的功能；** 我需要确保我的工作内容不会被你污染；
4. **在你执行代码遇到错误时，第一时间停下来向我询问；** 我需要确保你不会陷入死胡同；
5. **在你不知道、不确定、不清楚的任意时刻，告知我你的不确定性；** 我需要规避风险；
6. 每次执行写入操作之前，询问我是否需要使用git进行管理。

## TODO

### 模型层面
- [ ] 更新qwen2-audio模型文件 (@specforge/modeling/target/qwen2_audio.py)
- [ ] 更新draft model for qwen2-audio (@specforge/modeling/draft/qwen2_audio_eagle.py)
- [ ] 更新AudioEagle (@specforge/core/AudioEagle3.py)

### 数据层面
- [ ] 更新语音数据dataset (@specforge/data/)
- [ ] 更新语音数据dataloader (@specforge/data/)

### 运行脚本
- [ ] 更新运行脚本 (@scripts/train_audio_eagle3_online.py)