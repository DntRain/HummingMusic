# 协作规范

## 分支管理

### 分支命名规范

```
feature/<模块名>-<功能描述>    # 新功能
fix/<问题描述>                 # 问题修复
docs/<文档描述>                # 文档更新
```

示例：
- `feature/quantizer-bilstm-crf`
- `feature/audio-processing-crepe-integration`
- `fix/renderer-fluidsynth-fallback`
- `docs/readme-dataset-instructions`

### 各成员开发分支

| 成员 | 分支 | 负责模块 |
|------|------|---------|
| 成员A | `feature/audio-processing` | `src/audio_processing.py` |
| 成员B | `feature/quantizer` | `src/quantizer.py` |
| 成员C | `feature/style-transfer` | `src/style_transfer.py` |
| 成员D | `feature/gradio-ui` | `src/app.py`, `src/renderer.py` |

### 工作流程

1. 从 `main` 拉取最新代码
2. 在自己的 feature 分支上开发
3. 完成后推送到远程，提交 Pull Request 到 `main`
4. 等待 CI 通过 + 至少一人 code review
5. review 通过后合并

```bash
# 日常开发流程
git checkout main
git pull origin main
git checkout feature/your-branch
git merge main                    # 合并最新 main 到开发分支
# ... 开发 ...
git add <files>
git commit -m "[模块名] feat: 你的描述"
git push origin feature/your-branch
# 在 GitHub 上创建 PR
```

## Commit 信息格式

```
[模块名] <类型>: <描述>
```

### 模块名

`audio` | `quantizer` | `style` | `renderer` | `app` | `test` | `config` | `init` | `docs` | `ci`

### 类型

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修复 bug |
| `refactor` | 重构（不改变功能） |
| `test` | 添加或修改测试 |
| `docs` | 文档修改 |
| `chore` | 构建/配置等杂务 |

### 示例

```
[quantizer] feat: 添加BiLSTM-CRF模型类定义
[audio] fix: 修复短NaN段插值边界条件
[app] feat: 添加Piano Roll可视化组件
[style] refactor: 抽取和弦推断为独立函数
[test] feat: 补充renderer模块边界测试
[ci] chore: 添加flake8代码风格检查
```

## Pull Request 要求

### 合并前必须满足

- [ ] CI 全部通过（pytest + flake8）
- [ ] 至少一位其他成员 review 并 approve
- [ ] 不破坏现有接口（`interfaces.py` 的修改需全员讨论）
- [ ] 新功能附带对应的单元测试

### PR 描述模板

```markdown
## 变更说明
简要描述改了什么，为什么改。

## 测试
- [ ] 本地 pytest 通过
- [ ] 新增/修改了哪些测试

## 关联
关联的 issue 编号（如有）。
```

## 接口修改规范

`src/interfaces.py` 是模块间的契约，修改需谨慎：

1. 在群里提出修改需求并讨论
2. 创建专门的 PR，标题以 `[interfaces]` 开头
3. **全员 review** 后才可合并
4. 合并后各模块同步适配

## 代码风格

- 遵循 PEP 8（行长不限制，flake8 已忽略 E501）
- 函数必须写 docstring，说明输入输出格式
- 类型注解尽量完整
- 中文注释，英文变量名
