# Stream MoE 项目交接摘要

## 1. 当前工作区与代码基线

本项目已经从旧 Demo 工作区转向原版 Stream MoE 分支继续开发。

- 旧 Demo 工作区：`C:\Users\admin\Desktop\llama.cpp(7.16)`
- 新 Stream MoE 工作区：`C:\Users\admin\Desktop\llama.cpp-stream-moe`
- 本地分支：`stream-moe-baseline`
- 原作者仓库：`https://github.com/freedomljc/llama.cpp.git`
- 原作者分支：`feat/moe-streaming-core`
- 建立新工作区时的基线提交：`1248fd8fa8cfebaece5ea992e4d951c1e18bb9d5`
- 用户测试仓库：`https://github.com/544103252/llama-stream-moe-test.git`
- 对应 remote 名称：`streamrepo`

新对话开始时必须先执行：

```powershell
Get-Location
git branch --show-current
git rev-parse --short HEAD
git status --short
git remote -v
```

确认操作对象是 `llama.cpp-stream-moe`，不要继续修改旧 Demo 工作区。

## 2. 旧 Demo 已完成的研究成果

旧 Demo 曾完成或验证以下内容：

1. 将固定的48层、256专家、top-k=8等参数改为从模型信息动态获取。
2. 每层专家缓存数量可通过命令行配置。
3. 适配Qwen3.5MoE 35B和122B，并处理35B的MTP尾层加载问题。
4. 将原先只支持 `-ub 1` 的逻辑扩展为多token处理。
5. 建立GGUF专家切片地址表，并按layer/expert/kind读取gate/up/down权重。
6. 将部分缓存调度从Vulkan文件中分离到通用模块。
7. 尝试使用eval callback在argsort节点形成观察断点。
8. 实验过GPU侧缓存plan、expert_map、mapped_topk及命中/缺失统计。
9. 减少Vulkan同步过程中不必要的cleanup后，decode速度显著恢复。

这些研究用于理解机制，但旧 Demo 的实现方式不应整体复制到Stream MoE。

## 3. 原版 Stream MoE 当前架构

Stream MoE在 `src/llama-graph.cpp` 中把路由结果接入CPU自定义算子：

```cpp
ids_cont = ggml_cont(ctx0, selected_experts);
ids_gemm = ggml_map_custom1(ctx0, ids_cont, llama_moe_stream_remap, 1, msl);
```

当前主要执行路径：

```text
GPU计算router/top-k
  -> 将top-k专家ID复制到CPU
  -> CPU执行llama_moe_stream_remap
       - 统计本轮所需专家
       - 查询expert_slot
       - 判断hit/miss
       - 按热度衰减+LRU选择victim
       - 缺失时安排I/O worker读取GGUF
       - 等待需求专家resident
       - 输出mapped slot ID
  -> 将mapped slot ID复制回GPU
  -> GPU使用缓存权重执行MoE
```

主要文件：

- `src/llama-moe-stream.cpp`
- `src/llama-moe-stream.h`
- `src/llama-graph.cpp`
- `src/llama-model.cpp`
- `common/arg.cpp`

Stream MoE已有能力：

- 设备侧固定专家缓存张量；
- CPU侧 `expert_slot` 驻留映射；
- 热度衰减+LRU淘汰；
- 多I/O线程异步读取缺失专家；
- Linux O_DIRECT，Windows退化为普通读取；
- wave-partitional prefill；
- 从模型元信息和GGUF offset建立读取关系；
- 标准scheduler计算图分配，不依赖旧Demo运行时修改node src。

## 4. 哪些旧Demo改动不需要迁移

以下内容在Stream MoE中已经存在或已被更合理的实现替代：

- 固定层数/专家数等参数解耦；
- `-ub 1`限制修复；
- MTP尾层的旧版特殊处理；
- GGUF专家offset和切片读取；
- 命令行缓存容量设置；
- 专家缓存张量创建；
- 多token prefill处理；
- 简单淘汰策略；
- provider注册链；
- 按节点相对编号或名称搜索up/gate/down；
- 运行时修改 `node->src`；
- Vulkan内部强制提交、callback分图和cleanup补丁。

特别不要直接迁移：

- Qwen3.5专属provider；
- Vulkan文件中的模型语义和缓存策略硬编码；
- 依赖计算图节点相对位置的逻辑；
- 旧Demo的 `next_map`/commit实现；
- 为旧callback架构编写的同步和cleanup修补。

## 5. 值得在Stream MoE中重新实现的优化思想

核心方向是减少每个MoE层固定存在的GPU->CPU->GPU边界。

当前：

```text
GPU top-k
  -> CPU llama_moe_stream_remap
  -> GPU mapped_topk/MoE
```

目标：

```text
GPU top-k
  -> GPU查询expert_map
  -> GPU生成mapped_topk和miss信息
  -> hit时直接继续GPU MoE
  -> miss时仅通知CPU读取缺失专家到指定slot
```

值得保留的思想：

1. GPU侧 `expert_map`（expert ID -> cache slot）。
2. GPU侧生成 `mapped_topk`。
3. GPU侧完成hit/miss探测。

4. miss时只向CPU返回精简加载计划，而不是完整argsort。
5. hit时避免无意义的map复制、commit和专家上传。
6. 后端无关的算子接口，CPU实现作为fallback，Vulkan提供GPU实现。
7. 清晰的缓存命中率、缺失专家数和加载停顿统计。

## 6. 关键技术限制

当前 `llama_moe_stream_remap` 通过 `ggml_map_custom1` 创建，属于CPU自定义算子。因此scheduler必然形成：

```text
Vulkan子图 -> CPU子图 -> Vulkan子图
```

仅把缓存判断代码改写成Vulkan C++代码，不等于成为真正的GPU算子。

建议分阶段实现：

1. 为“专家ID映射到缓存槽位”定义正式、后端可实现的算子契约。
2. 保留当前CPU remap作为fallback，确保行为和精度不变。
3. 为Vulkan实现GPU kernel，并在GPU维护expert_map/mapped_topk。
4. 对比CPU和Vulkan输出，保证slot映射完全一致。
5. 再研究cache hit不返回CPU的快速路径。

第5步最困难。普通静态计算图和eval callback不能依据GPU运行结果动态跳过CPU：callback一旦设为观察断点就一定会同步返回CPU。真正的miss-only CPU路径需要条件执行、重算方案、GPU驱动I/O，或scheduler层新增机制，不能靠简单移动代码完成。

## 7. 当前性能基线

测试环境：Windows + Vulkan + Qwen3.5-35B-A3B Q4_K_M，缓存180专家/层。

示例命令：

```powershell
& "D:\llama-cpp\LT\LLM_stream_moe\build\bin\Release\llama-cli.exe" `
  -m "D:\llama-cpp\models\Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf" `
  -c 8192 -ngl 999 `
  --reasoning off `
  --moe-stream-cache 180s `
  --moe-stream-io-threads 4 `
  --no-mmap
```

观察结果：

- 第一轮Prompt约64 t/s，Generation约27 t/s；
- 后续Prompt约161 t/s，Generation约27.4 t/s；
- 重复相似生成时decode曾从约27逐步提升到34、36 t/s；
- 专用GPU内存约16.3 GiB，总GPU内存约16.8 GiB。

速度逐渐上升很可能由以下共同造成：

- 180个专家槽逐渐形成稳定工作集；
- 热度统计与LRU逐渐稳定；
- Windows文件页缓存预热；
- Vulkan管线与GPU频率预热。

内置Stream统计由 `llama_moe_stream::print_stats()` 输出，包含：

- remap calls；
- expert hits/misses；
- hit rate；
- load stall；
- wave/preload统计。

需要正常退出且INFO日志开启。可显式使用：

```text
--perf -lv 3 --log-file .\stream-moe.log
```

## 8. Stream MoE命令行参数语义

- `--moe-stream-cache 180s`：每个streamed MoE层精确缓存180个专家。
- `--moe-stream-cache 40`：总缓存预算40 GiB，再换算为每层slot数。
- `--moe-stream`且不指定cache：自动使用 `clamp(2*n_expert_used, 16, n_expert)`。
- `--moe-stream-io-threads 4`：4个CPU I/O worker并行读取缺失专家，不是推理线程或GPU线程。
- 缓存slot数达到 `n_expert` 时会禁用streaming并完整加载专家；测试streaming时不要使用256s。
- 关闭思考使用 `--reasoning off`，不是旧版的 `--reasoning-budget 0`。

## 9. 推荐的下一步

下一步不要立即移植旧Demo。先围绕 `llama_moe_stream_remap` 建立清晰边界：

1. 明确其输入：原始top-k expert IDs。
2. 明确其输出：cache slot IDs。
3. 列出CPU实现当前读取和修改的全部状态。
4. 区分纯映射部分、淘汰策略部分、GGUF I/O部分。
5. 设计后端算子输入/输出和CPU fallback。
6. 先完成结果等价性测试，再编写Vulkan kernel。

在任何代码变动前，应向用户说明改动范围、前后流程和可验证方式。每次只进行一个小阶段，远程验证后再继续。

## 10. Git工作方式

Stream工作区使用独立远端，不再使用旧Demo的 `mytest LT:main`。

阶段性新提交：

```powershell
git add <files>
git commit -m "由用户本人编写的提交说明"
git push streamrepo stream-moe-baseline:main
```

同一阶段继续覆盖最后一次提交：

```powershell
git add <files>
git commit --amend --no-edit
git push --force-with-lease streamrepo stream-moe-baseline:main
```

远程测试机若对应远端main被amend强推，在确认测试工作区没有需要保留的本地修改后使用：

```powershell
git fetch origin
git reset --hard origin/main
```

不要由AI自动提交或推送；每次commit/push均由用户明确确认并执行。
