import { hasMath, rich } from './lib/math';
/** English / Chinese strings. Keys used in index.html via data-i18n / data-i18n-html. */
export type Lang = 'en' | 'zh';
type Params = Record<string, string | number>;
type Entry = { en: string; zh: string };

const D = {
  'doc.title': {
    en: 'Prefix-Tuning for Arbitrary Output Sequences on Pretrained Transformers · Attention Matching',
    zh: 'Prefix-Tuning for Arbitrary Output Sequences on Pretrained Transformers · Attention Matching',
  },
  'lang.switch': { en: '中文', zh: 'English' },
  'hdr.code': { en: 'Code', zh: '代码' },
  'paper.title': {
    en: 'Prefix-Tuning for Arbitrary Output Sequences on Pretrained Transformers',
    zh: 'Prefix-Tuning for Arbitrary Output Sequences on Pretrained Transformers',
  },
  'paper.venue': { en: 'Accepted at NeurIPS 2026 · Come find us at the Sydney venue', zh: '已被 NeurIPS 2026 接收 · 欢迎到悉尼会场与我们交流' },
  'paper.pdf': { en: 'Paper (PDF)', zh: '论文 PDF' },

  // intro
  'intro.sub': {
    en: 'Steering a pretrained transformer with a prefix that is solved, not trained',
    zh: 'prefix 由求解得到，而非训练得到',
  },
  'intro.p2': {
    en: 'We evaluate the method on two models with different attention designs. The table reports the fraction of input pairs for which the model with the prefix generates exactly $y$, for outputs of $k=1$, $5$ and $10$ tokens.',
    zh: '我们在两种 attention 结构不同的模型上评估了该方法。下表给出接入 prefix 后模型恰好生成 $y$ 的输入对所占比例，输出长度 $k$ 分别为 $1$、$5$、$10$ 个 token。',
  },
  'res.model': { en: 'Model', zh: '模型' },
  'res.attn': { en: 'Attention', zh: 'Attention' },
  'res.gpt2': { en: 'multi-head, 12 layers', zh: 'multi-head attention，12 层' },
  'res.qwen': { en: 'grouped-query, QK-Norm, 36 layers', zh: 'grouped-query attention（GQA），QK-Norm，36 层' },
  'res.note': {
    en: 'GPT-2: each rate is the mean of Table 1 in the paper over its 20 settings (dataset × input length) for that $k$, with $r=90$. Qwen3-4B: $r=10$, on the four datasets rebuilt for the Qwen3 tokenizer with 50 sentences per bucket, so no pair is excluded; all 60 groups reach 100%, and even without repetition ($r=1$) the overall rate is 99.65%.',
    zh: 'GPT-2：每个比例为论文 Table 1 中该 $k$ 下 20 个设置（数据集 × 输入长度）的平均值，$r=90$。Qwen3-4B：$r=10$，四个数据集按 Qwen3 的 tokenizer 重建，每桶 50 句，没有排除任何输入对；全部 60 组都达到 100%，即使不重复（$r=1$），总体成功率也有 99.65%。',
  },
  'fig.src': {
    en: 'source run',
    zh: 'source 运行',
  },
  'fig.solve': {
    en: 'solved in every head',
    zh: '在每个 head 上求解',
  },
  'fig.prefix': {
    en: 'prefix',
    zh: 'prefix',
  },
  'fig.new': { en: 'new input', zh: '新输入' },
  'fig.with': { en: 'with prefix', zh: '有 prefix' },
  'fig.without': { en: 'without prefix', zh: '无 prefix' },
  'fig.cap': {
    en: 'Top: the model continues $x\'$ with $y$. Middle: from that one run, a prefix $P=(\\tilde K,\\tilde V)$ is solved in every head of every layer, from $S\\tilde V=R\\,V_{\\mathrm{src}}$. Bottom: on a different input $x$, the model with $P$ writes $y$; without it, the model writes its own continuation.',
    zh: '上：模型将 $x\'$ 续写为 $y$。中：由这一次运行，在每一层的每个 head 上由 $S\\tilde V=R\\,V_{\\mathrm{src}}$ 解出 prefix $P=(\\tilde K,\\tilde V)$。下：换用输入 $x$ 后，带 $P$ 的模型写出 $y$，不带 $P$ 的模型则写出它自己的续写。',
  },

  // viz
  'ctl.src': {
    en: 'Opening line',
    zh: '开头',
  },
  'ctl.k': {
    en: 'Output tokens',
    zh: '输出 token 数',
  },
  'ctl.termBad': { en: 'Your input must end with “{a}”, like the source.', zh: '你的输入需与 source 一样以“{a}”结尾。' },
  'ctl.fix': { en: 'Fix it', zh: '自动修正' },
  'demo.badge': { en: 'Simulated', zh: '模拟数据' },
  'demo.tip': {
    en: 'No GPU is connected. Model outputs and attention maps are simulated in the browser; the [S, W and R figure](#fig-svr) further down applies the real equations to random vectors.',
    zh: '当前未连接 GPU：模型输出与注意力图均在浏览器中模拟生成；下文的 [S、W、R 图](#fig-svr)按真实公式计算，但输入为随机向量。',
  },
  'offline': { en: 'The GPU server is offline.', zh: 'GPU 服务器已下线。' },
  'offline.use': { en: 'Use simulated data', zh: '改用模拟数据' },
  'mode.off': { en: 'Without prefix', zh: '无 prefix' },
  'mode.on': { en: 'With prefix', zh: '有 prefix' },
  'nav.layer': { en: 'Layer', zh: '层' },
  'nav.head': { en: 'Head', zh: 'head' },
  'nav.r': { en: 'Repeats r', zh: '重复次数 r' },
  'st.solving': {
    en: 'Creating the prefix…',
    zh: '正在生成 prefix…',
  },
  'st.generating': {
    en: 'Thinking…',
    zh: '思考中…',
  },
  'st.queue': { en: 'Waiting for the GPU ({n} ahead)…', zh: '排队中，前方还有 {n} 个请求…' },
  'err.bad_input': {
    en: 'The text box is empty.',
    zh: '输入框不能为空。',
  },
  'err.too_long': {
    en: 'That is too long: the limit is {maxN} tokens.',
    zh: '输入过长：最多 {maxN} 个 token。',
  },
  'err.prefix_not_found': { en: 'The prefix expired. Press Run again.', zh: 'prefix 已过期，请重新运行。' },
  'err.gen_not_found': { en: 'This run expired. Press Run again.', zh: '本次运行已过期，请重新运行。' },
  'err.other': { en: 'Something went wrong: {msg}', zh: '出现错误：{msg}' },

  // article
  'art.fig2.cap': {
    en: '$S$, $W$ and $R$ for layer {l}, KV head {kv} of the [demo above](#demo). Rows of $S$ and $R$ are the $m=Gk$ stacked queries, grouped by query head (thin lines every $k$ rows); within a group, row $t$ is the query that writes $y_{t+1}$. Columns of $S$ are the $m$ prefix slots, in the same order. Columns of $W$ and $R$ are the source positions: $x\'_1,\\dots,x\'_p$, then $y_1,\\dots,y_{k-1}$ after the vertical line. $\\kappa(S) = {cond}$, relative residual $\\|SW-R\\|/\\|R\\| = {res}$.',
    zh: '[上方演示](#demo)中第 {l} 层、第 {kv} 个 KV head 的 $S$、$W$ 与 $R$。$S$ 和 $R$ 的行是堆叠起来的 $m=Gk$ 个 query，按 query head 分组（每 $k$ 行一条细线）；组内第 $t$ 行是写出 $y_{t+1}$ 的那个 query。$S$ 的列是 $m$ 个 prefix slot，顺序与行相同。$W$ 和 $R$ 的列是 source 的各个位置：先是 $x\'_1,\\dots,x\'_p$，竖线之后是 $y_1,\\dots,y_{k-1}$。$\\kappa(S) = {cond}$，相对残差 $\\|SW-R\\|/\\|R\\| = {res}$。',
  },
  'svr.empty': {
    en: 'Compute a prefix in Step 2 of the demo above to see this figure.',
    zh: '在上方演示的第 2 步计算 prefix 后，这里会显示该图。',
  },
  'art.fig2.S': {
    en: '$S$ · prefix attention',
    zh: '$S$ · prefix 注意力',
  },
  'art.fig2.V': {
    en: '$W = S^{-1}R$',
    zh: '$W = S^{-1}R$',
  },
  'art.ax.S': { en: 'rows: queries · cols: prefix slots', zh: '行：query · 列：prefix slot' },
  'art.ax.W': { en: 'rows: prefix slots · cols: source positions', zh: '行：prefix slot · 列：source 位置' },
  'art.ax.R': { en: 'rows: queries · cols: source positions', zh: '行：query · 列：source 位置' },
  'art.fig2.R': {
    en: '$R$ · source attention',
    zh: '$R$ · source 的注意力',
  },
  'foot': {
    en: 'Code and experiments: <a href="https://github.com/wzii/prefix_tuning">github.com/wzii/prefix_tuning</a>. When connected to the GPU server, the demo runs Qwen/Qwen3-4B with float32 weights and a float64 solver.',
    zh: '代码与实验：<a href="https://github.com/wzii/prefix_tuning">github.com/wzii/prefix_tuning</a>。接入 GPU 服务器后，在线演示使用 Qwen/Qwen3-4B，模型权重为 float32，求解精度为 float64。',
  },
  'nav.overview': { en: 'Overview', zh: '概述' },
  'nav.cite': { en: 'Citation', zh: '引用' },
  'cite.p': { en: 'If you find this work useful, please cite our paper:', zh: '如果这项工作对你有帮助，欢迎引用我们的论文：' },
  'cite.copy': { en: 'Copy', zh: '复制' },
  'cite.copied': { en: 'Copied', zh: '已复制' },
  'nav.demo': { en: 'Demo', zh: '在线演示' },
  'nav.method': { en: 'Method', zh: '方法' },
  'nav.results': { en: 'Results', zh: '实验结果' },
  'nav.tryDemo': {
    en: 'Try it yourself',
    zh: '亲自试一试',
  },
  'demo.h': {
    en: 'Try it yourself: Force an LLM to generate your target sentence with a computed prefix',
    zh: '亲自试一试：用计算出的 prefix 逼大模型说出你的目标句子',
  },
  'make.h': {
    en: 'Step 1: Pick your target output sentence',
    zh: '第 1 步：选择你的目标输出句子',
  },
  'make.p': {
    en: 'Choose the sentence the model will be forced to say.',
    zh: '选一句你想逼模型说出的话。',
  },
  'make.dl': {
    en: 'Download',
    zh: '下载',
  },
  'run.upload': {
    en: 'upload a file',
    zh: '上传文件',
  },
  'run.remove': { en: 'Remove', zh: '移除' },
  'run.need': {
    en: 'Compute a prefix in Step 2 first.',
    zh: '请先在第 2 步计算 prefix。',
  },
  'run.loadedFrom': { en: 'from your upload', zh: '来自你上传的文件' },
  'card.meta': { en: '{layers} layers × {kv} KV heads × {m} slots · {size}', zh: '{layers} 层 × {kv} 个 KV head × {m} 个 slot · {size}' },
  'st.uploading': { en: 'Checking the file…', zh: '正在校验文件…' },
  'results.h': { en: 'Results', zh: '实验结果' },
  'mech.src': {
    en: 'Source run',
    zh: 'source 运行',
  },
  'mech.new': {
    en: 'New input + prefix',
    zh: '新输入 + prefix',
  },
  'mech.out': { en: 'attention output', zh: 'attention 输出' },
  'mech.same': { en: 'equal', zh: '相等' },
  'mech.cap': {
    en: 'Bar height is the attention weight of one query $q_t$. Left: in the source run, the weights $R_{t,j}$ are spread over $x\'$ and $y$, and the head outputs $a_t$. Right: with the prefix, almost all of the weight, $S_{t,i}$, is on the slot solved for this query. Because $\\tilde V$ solves $S\\tilde V=R\\,V_{\\mathrm{src}}$, this head outputs $a_t$ as well.',
    zh: '柱高表示某个 query $q_t$ 的 attention 权重。左：source 运行时，权重 $R_{t,j}$ 分布在 $x\'$ 与 $y$ 上，head 输出 $a_t$。右：接入 prefix 后，几乎全部权重 $S_{t,i}$ 落在为这个 query 求解的 slot 上。由于 $\\tilde V$ 满足 $S\\tilde V=R\\,V_{\\mathrm{src}}$，这个 head 的输出同样是 $a_t$。',
  },
  'err.bad_file': {
    en: 'This file cannot be used. {msg}',
    zh: '该文件无法使用：{msg}',
  },
  'make.tabEx': {
    en: 'Examples',
    zh: '示例',
  },
  'make.tabCustom': {
    en: 'Write your own',
    zh: '自己写一句',
  },
  'make.want': {
    en: 'Your sentence',
    zh: '你的句子',
  },
  'make.find': {
    en: 'Use this sentence',
    zh: '就用这一句',
  },
  'make.finding': {
    en: 'Checking…',
    zh: '检查中…',
  },
  'make.toManual': {
    en: 'Advanced: start from an opening line instead',
    zh: '高级：改用一句开头',
  },
  'solve.h': {
    en: 'Step 2: Compute a prefix',
    zh: '第 2 步：计算 prefix',
  },
  'solve.p': {
    en: 'The prefix is not trained. Your sentence is turned into one small linear system per attention head, and solving them gives the prefix.',
    zh: 'prefix 不需要训练：你的句子会在每个 attention head 上变成一个小的线性方程组，解出这些方程组就得到 prefix。',
  },
  'solve.btn': {
    en: 'Compute prefix',
    zh: '计算 prefix',
  },
  'solve.busy': {
    en: 'Computing…',
    zh: '计算中…',
  },
  'solve.done': {
    en: 'Computed ✓',
    zh: '已完成 ✓',
  },
  'solve.need': {
    en: 'Pick a target sentence in Step 1 first.',
    zh: '请先在第 1 步选择目标句子。',
  },
  'an.idle': {
    en: 'Press the button to build the linear systems from your sentence and solve them.',
    zh: '点击按钮，用你的句子构造线性方程组并求解。',
  },
  'an.s1': {
    en: '① The source run: after a hidden prompt $x\'$, the model writes your sentence $y$ by itself.',
    zh: '① source 运行：在隐藏 prompt $x\'$ 之后，模型自己写出你的句子 $y$。',
  },
  'an.s1manual': {
    en: '① The source run: after your opening line $x\'$, the model writes $y$.',
    zh: '① source 运行：在你写的开头 $x\'$ 之后，模型写出 $y$。',
  },
  'an.s2': {
    en: '② Each output token $y_t$ is written by one query in every query head, and each query adds one row to the linear system $S\\,W = R$.',
    zh: '② 每个输出 token $y_t$ 在每个 query head 中由一个 query 写出，每个 query 给线性方程组 $S\\,W = R$ 添上一行。',
  },
  'an.s3': {
    en: '③ Solve for $W$ and set the prefix values to $\\tilde V = W\\,V_{\\mathrm{src}}$, in every KV head of every layer.',
    zh: '③ 解出 $W$，令 prefix 的 value 为 $\\tilde V = W\\,V_{\\mathrm{src}}$；每一层的每个 KV head 都解一次。',
  },
  'an.hidden': {
    en: 'hidden prompt · {p} tokens',
    zh: '隐藏 prompt · {p} 个 token',
  },
  'an.shown': {
    en: 'Shown: layer {l}, KV head {kv}. {m} rows = {G} query heads × {k} tokens; {T} source positions.',
    zh: '图中：第 {l} 层、第 {kv} 个 KV head。{m} 行 = {G} 个 query head × {k} 个 token；{T} 个 source 位置。',
  },
  'an.count': {
    en: '{n} linear systems ({L} layers × {H} KV heads) solved in {ms} ms. Largest condition number $\\kappa(S) = {cond}$; largest relative residual {res}.',
    zh: '共 {n} 个线性方程组（{L} 层 × {H} 个 KV head），用时 {ms} 毫秒。最大条件数 $\\kappa(S) = {cond}$，最大相对残差 {res}。',
  },
  'file.empty': {
    en: 'The computed prefix will appear here as a .safetensors file.',
    zh: '计算出的 prefix 会以 .safetensors 文件的形式出现在这里。',
  },
  'file.solving': {
    en: 'Solving…',
    zh: '求解中…',
  },
  'file.have': {
    en: 'Already have a prefix file?',
    zh: '已经有 prefix 文件？',
  },
  'insp.open': {
    en: 'Inspect file',
    zh: '查看文件内容',
  },
  'insp.close': {
    en: 'Hide details',
    zh: '收起',
  },
  'insp.loading': {
    en: 'Reading the file…',
    zh: '正在读取文件…',
  },
  'insp.bad': {
    en: 'This file cannot be read here: {msg}',
    zh: '无法在这里读取该文件：{msg}',
  },
  'insp.header': {
    en: 'Header metadata',
    zh: '文件头元数据',
  },
  'insp.m.format': {
    en: 'file format and version',
    zh: '文件格式与版本',
  },
  'insp.m.model': {
    en: 'the model the prefix is for',
    zh: 'prefix 适用的模型',
  },
  'insp.m.p': {
    en: 'length of the source, in tokens',
    zh: 'source 的长度（token 数）',
  },
  'insp.m.k': {
    en: 'length of the target sentence, in tokens',
    zh: '目标句子的长度（token 数）',
  },
  'insp.m.last_token': {
    en: 'last token of the source; your message must end with it',
    zh: 'source 的最后一个 token；你的消息必须以它结尾',
  },
  'insp.m.expected_continuation': {
    en: 'the target sentence, as tokens',
    zh: '目标句子（按 token 切分）',
  },
  'insp.m.signature': {
    en: 'the server’s signature, so it can take the file back without checking it again',
    zh: '服务器的签名，再次上传时可免于重新校验',
  },
  'insp.tensors': {
    en: 'Tensors',
    zh: '张量',
  },
  'insp.tsum': {
    en: '{n} tensors · {dtype} · {size}. Each layer has a key tensor and a value tensor, each of shape [{kv}, {m}, {hd}] = [KV heads, slots, head dimension].',
    zh: '共 {n} 个张量 · {dtype} · {size}。每层一个 key 张量和一个 value 张量，形状都是 [{kv}, {m}, {hd}]，即 [KV head，slot，head 维度]。',
  },
  'insp.all': {
    en: 'List all {n} tensors',
    zh: '列出全部 {n} 个张量',
  },
  'insp.peek': {
    en: 'Look inside a tensor',
    zh: '查看张量的数值',
  },
  'insp.keys': {
    en: 'Keys',
    zh: 'Key',
  },
  'insp.values': {
    en: 'Values',
    zh: 'Value',
  },
  'insp.rows': {
    en: 'Each row is one slot, aimed at one query: a query head writing one output token. Columns are the {hd} coordinates. Hover a cell for details. At run time each slot is repeated $r$ times; the file stores it once.',
    zh: '每一行是一个 slot，对准一个 query：某个 query head 写出某个输出 token 时的那个 query。列是 {hd} 个坐标。把鼠标移到格子上可查看详情。运行时每个 slot 会重复 $r$ 次，文件里只存一份。',
  },
  'insp.stats': {
    en: 'min {min} · max {max} · mean |x| {mean}',
    zh: '最小值 {min} · 最大值 {max} · 平均 |x| {mean}',
  },
  'insp.tip': {
    en: 'slot {i}: query head {h}, writing “{tok}” · dim {j} = {v}',
    zh: 'slot {i}：query head {h}，写出“{tok}” · 第 {j} 维 = {v}',
  },
  'insp.missing': {
    en: '{name} is not in this file.',
    zh: '文件中没有 {name}。',
  },
  'make.missT': {
    en: 'We can’t force the model to say exactly that. The closest we can force is:',
    zh: '没法逼模型说出一模一样的话，最接近的是这一句：',
  },
  'make.asked': {
    en: 'You wanted',
    zh: '你想要的',
  },
  'make.found': {
    en: 'We can force',
    zh: '能逼它说的',
  },
  'make.accept': {
    en: 'Use this one',
    zh: '就用这一句',
  },
  'make.manual': {
    en: 'Start from an opening line',
    zh: '改用一句开头',
  },
  'make.manualP': {
    en: 'Type an opening line. Whatever the model writes after it becomes the sentence.',
    zh: '输入一句开头，模型接着写出的话就是那句话。',
  },
  'make.back': { en: '← back', zh: '← 返回' },
  'rej.empty': {
    en: 'Type the sentence the model should be forced to say.',
    zh: '请输入要逼模型说出的话。',
  },
  'rej.too_long': { en: 'That is too long. Keep it within {maxK} tokens.', zh: '内容过长，请控制在 {maxK} 个 token 以内。' },
  'rej.unsafe': { en: 'We can’t make the model say that. Please try something else.', zh: '出于安全考虑，我们无法让模型生成这句话，请换一句试试。' },
  'rej.pii': { en: 'Please leave out links, email addresses and phone numbers.', zh: '请勿包含链接、邮箱或电话号码。' },
  'rej.chars': { en: 'Please use plain text.', zh: '请只输入普通文本。' },
  'method.why.h': {
    en: 'Why there has to be a source',
    zh: '为什么必须有 source',
  },
  'method.why': {
    en: 'The prefix copies attention from a run the model actually made: $q_t$, $R$ and $V_{\\mathrm{src}}$ all come from running $T$ on $x\'$ followed by $y$. A sentence the model would never write has no such run, so attention matching does not apply. This is why [the demo](#demo) first searches for a hidden prompt after which the model writes your sentence, and why every message ends with a line break: every hidden prompt ends with one, so $x_n=x\'_p$ always holds.',
    zh: 'prefix 复制的是模型一次真实运行中的 attention：$q_t$、$R$ 和 $V_{\\mathrm{src}}$ 都来自让 $T$ 在 $x\'$ 后接 $y$ 上的运行。模型永远不会写出的句子，就不存在这样一次运行，attention matching 也就无法适用。因此[演示](#demo)会先寻找一个隐藏 prompt，使模型在它之后写出你的句子；每条消息末尾也因此都加一个换行：所有隐藏 prompt 都以换行结尾，条件 $x_n=x\'_p$ 便总能满足。',
  },
  'make.why': {
    en: 'Why not any sentence? The prefix can only force a sentence the model would say on its own after some opening line. <a href="#why-source">How it works</a>',
    zh: '为什么不能是任意一句？prefix 只能逼模型说出它在某个开头之后本来就会说的话。<a href="#why-source">了解原理</a>',
  },
  'mech.outFull': {
    en: 'attention output',
    zh: 'attention 输出',
  },
  'file.format': {
    en: 'This is not a .safetensors file.',
    zh: '这不是 .safetensors 文件。',
  },
  'file.model': {
    en: 'This prefix was made for a different model.',
    zh: '这个 prefix 是为其他模型生成的。',
  },
  'file.shape': {
    en: 'The tensors in this file do not have the expected shape.',
    zh: '文件中张量的形状与预期不符。',
  },
  'file.last_token': {
    en: 'The file does not record which token the source ends with.',
    zh: '文件中没有记录 source 的末 token。',
  },
  'aria.rPrev': {
    en: 'Fewer repeats',
    zh: '减少重复次数',
  },
  'aria.rNext': {
    en: 'More repeats',
    zh: '增加重复次数',
  },
  'aria.layerPrev': {
    en: 'Previous layer',
    zh: '上一层',
  },
  'aria.layerNext': {
    en: 'Next layer',
    zh: '下一层',
  },
  'aria.headPrev': {
    en: 'Previous head',
    zh: '上一个 head',
  },
  'aria.headNext': {
    en: 'Next head',
    zh: '下一个 head',
  },
  'nav.limits': {
    en: 'Limitations',
    zh: '局限',
  },
  'hero.expert': {
    en: 'More formally, our work establishes the theoretical expressiveness of prefix-tuning as a parameter-efficient fine-tuning method (like LoRA), supporting its usefulness.',
    zh: '更正式地说，我们的工作确立了 prefix-tuning 作为一种参数高效微调方法（类似 LoRA）在理论上的表达能力，从而支持了它的实用价值。',
  },
  'mot.h': {
    en: 'Motivations',
    zh: '动机',
  },
  'mot.p': {
    en: 'Pretrained large language models often require fine-tuning to master specialized domains or follow specific instructions, but updating billions of parameters is prohibitively expensive. Parameter-efficient fine-tuning (PEFT) solves this by adapting models without altering their core weights.\n\nPrefix-tuning is a popular PEFT approach that builds on prompt engineering: while prompts manually select text tokens, prefix-tuning optimizes continuous vector embeddings by prepending a small sequence of learnable vectors.\n\nYet, a fundamental question remains: how expressive is this tiny set of parameters? This theoretical gap has persisted for realistic autoregressive Transformers. Here, we show that under mild conditions, a prefix can be explicitly constructed to make a pretrained Transformer generate any output sequence already reachable by the model.',
    zh: '预训练的大语言模型往往需要微调，才能掌握专业领域的知识或遵循特定指令，但更新数十亿个参数的代价高得难以承受。参数高效微调（PEFT）通过在不改动模型核心权重的情况下适配模型，解决了这一问题。\n\nPrefix-tuning 是一种流行的 PEFT 方法，它建立在提示工程的基础上：提示是人工挑选文本 token，而 prefix-tuning 则通过在前面拼接一小段可学习的向量，来优化连续的向量嵌入。\n\n然而，一个根本性的问题依然存在：这一小组参数的表达能力究竟有多强？对于现实中的自回归 Transformer，这一理论空白一直没有填补。在这里，我们证明了在温和的条件下，可以显式构造一个 prefix，使预训练的 Transformer 生成任何模型本来就能生成的输出序列。',
  },
  'm.am0': {
    en: 'To compute the desired prefix, we develop a novel technique called <strong>attention matching</strong>: when generating the target output during a source run, we collect the attention outputs at each head. In the subsequent run, we use the prefix to reproduce each of those attention outputs.',
    zh: '为了计算出所需的 prefix，我们提出了一种名为 <strong>attention matching</strong> 的新技术：在 source 运行中生成目标输出时，我们收集每个 head 的注意力输出；在随后的运行中，我们用 prefix 复现每一个这样的注意力输出。',
  },
  'm.rough': {
    en: 'Here is what we achieve, roughly speaking: given a target output sequence, we show that as long as it is within the range of the LLM (i.e., a source input for it exists), we can compute a prefix (i.e., solve for $\\tilde{k}_i$ and $\\tilde{v}_i$) that will force the model to generate it, almost independent of what the user asks next.',
    zh: '粗略地说，我们的成果是：给定一个目标输出序列，只要它在大语言模型的可达范围之内（即存在一个能让模型输出它的 source 输入），我们就能计算出一个 prefix（即求解 $\\tilde{k}_i$ 和 $\\tilde{v}_i$），迫使模型生成这个序列，而且几乎与用户接下来问什么无关。',
  },
  'm.rem1': {
    en: '<strong>Remark 1.</strong> We have some freedom in choosing $S$, since it depends on the prefix keys $\\tilde K$. See [the next section](#qk-norm) and our paper for more details.',
    zh: '<strong>注 1.</strong> $S$ 的取值有一定的自由度，因为它取决于 prefix 的 key $\\tilde K$。更多细节请参见[下一节](#qk-norm)和我们的论文。',
  },
  'm.rem2': {
    en: '<strong>Remark 2.</strong> In the paper, we solve for the prefix embeddings rather than for $\\tilde V$ directly. Solving for $\\tilde V$ is sufficient, and arguably better conditioned numerically, if one only needs the prefix values.',
    zh: '<strong>注 2.</strong> 在论文中，我们求解的是 prefix 的嵌入（embedding），而不是直接求解 $\\tilde V$。如果只需要 prefix 的 value，直接求解 $\\tilde V$ 就足够了，而且在数值上可以说条件更好。',
  },
  'hero.analogy': {
    en: 'A language model reads your message and writes a reply. In online services, though, your message is not all it reads: it often also receives a hidden note telling it how to behave, like “Be polite and professional” or “Don’t say certain things to users.”',
    zh: '语言模型读你发来的消息，然后写一段回复。不过在线上服务中，它读到的不只是你的消息：它往往还会收到一张告诉它该怎么做的隐藏纸条，比如“请保持礼貌和专业”，或者“不要对用户说某些话”。',
  },
  'hero.do': {
    en: 'Now think of a sentence you want the model to say. We find a note that forces the model to say exactly that, no matter what the user asks.[1](#hero-note) Our note needs no training, and, thanks to the mathematics, the model can hardly refuse. The [demo below](#demo) runs on Qwen3-4B, an open language model with 4 billion parameters.',
    zh: '现在，想一句你希望模型说出的话。我们能找到一张纸条，逼模型一字不差地说出这句话，无论用户问什么。[1](#hero-note)这张纸条不需要训练，而且有数学作保证，模型几乎无法拒绝。[下面的演示](#demo)使用的是 Qwen3-4B，一个拥有 40 亿参数的开源语言模型。',
  },
  'hero.note': {
    en: '¹ Provided the model would say that sentence after some input, and your message ends with the same token as that input. The exact conditions are in the [Method section](#theorem).',
    zh: '¹ 前提是模型在某个输入之后本来就会说出这句话，并且你的消息与那个输入以同一个 token 结尾。确切条件见[“方法”一节](#theorem)。',
  },
  'hero.what': {
    en: 'A prefix is exactly this kind of hidden note, written in numbers rather than words. Normally, people find a useful one by trial and error, training it for hours.',
    zh: 'prefix 就是这样一张隐藏的纸条，只不过上面写的是数字而不是文字。通常，人们要反复试错、训练好几个小时，才能得到一张好用的纸条。',
  },
  'make.ready': {
    en: 'Target sentence: “{y}”',
    zh: '目标句子：“{y}”',
  },
  'make.readyManual': {
    en: 'Target: whatever the model writes in the {k} tokens after your opening line.',
    zh: '目标：模型在你的开头之后接着写出的 {k} 个 token。',
  },
  'chat.h': {
    en: 'Step 3: Hack the LLM generation with the computed prefix',
    zh: '第 3 步：用计算出的 prefix 劫持大模型的生成',
  },
  'chat.loaded': {
    en: 'Prefix loaded: {name}',
    zh: '已载入 prefix：{name}',
  },
  'chat.ph': {
    en: 'Type a prompt for the model…',
    zh: '输入给模型的 prompt…',
  },
  'chat.send': {
    en: 'Send',
    zh: '发送',
  },
  'chat.you': {
    en: 'You',
    zh: '你',
  },
  'chat.ok': {
    en: 'Qwen3-4B',
    zh: 'Qwen3-4B',
  },
  'chat.broken': {
    en: 'Qwen3-4B, hacked by the prefix',
    zh: '被 prefix 劫持的 Qwen3-4B',
  },
  'chat.hit': {
    en: 'said exactly the chosen sentence',
    zh: '说出的正是选定的那句话',
  },
  'chat.miss': {
    en: 'did not say the chosen sentence',
    zh: '没有说出选定的那句话',
  },
  'chat.info': {
    en: 'We quietly add a line break after every message. The method needs your message to end with the same token as the hidden prompt, and every hidden prompt ends with a line break. <a href="#why-source">Why</a>',
    zh: '我们会在每条消息后面悄悄加一个换行。这个方法要求你的消息与隐藏 prompt 以同一个 token 结尾，而所有隐藏 prompt 都以换行结尾。<a href="#why-source">原因</a>',
  },
  'aria.info': {
    en: 'About this demo',
    zh: '关于这个演示',
  },
  'chat.empty': {
    en: 'Send a prompt above. Both models get the same message.',
    zh: '在上方发送一条 prompt，两个模型收到的是同一条消息。',
  },
  'chat.needPrefix': {
    en: 'Compute a prefix in Step 2; it is loaded here automatically.',
    zh: '先在第 2 步计算 prefix，它会自动加载到这里。',
  },
  'win.plain': {
    en: 'without prefix',
    zh: '不加 prefix',
  },
  'win.hacked': {
    en: 'with the prefix',
    zh: '加上 prefix',
  },
  'method.h': {
    en: 'Method',
    zh: '方法',
  },
  'm.intro': {
    en: 'Notation. We consider a decoder-only transformer $T$ with $L$ layers. In one attention head of one layer, a query $q \\in \\mathbb{R}^{d}$ attends to keys $k_1,\\dots,k_N$ and values $v_1,\\dots,v_N$ of the visible tokens.',
    zh: '记号。考虑一个有 $L$ 层的 decoder-only Transformer $T$。在某一层的某个 head 中，query $q \\in \\mathbb{R}^{d}$ 关注可见 token 的 key $k_1,\\dots,k_N$ 与 value $v_1,\\dots,v_N$。',
  },
  'm.pt.h': {
    en: 'Prefix-tuning in one equation',
    zh: '用一个公式表示 prefix-tuning',
  },
  'm.pt': {
    en: 'Prefix-tuning prepends $m$ extra key/value pairs $(\\tilde k_i, \\tilde v_i)$ to every head, so the head computes\n\n$$\\operatorname{Attn}(q)=\\frac{\\sum_{i=1}^{m} e^{q^\\top \\tilde k_i/\\sqrt d}\\,\\tilde v_i+\\sum_{j=1}^{N} e^{q^\\top k_j/\\sqrt d}\\,v_j}{\\sum_{i=1}^{m} e^{q^\\top \\tilde k_i/\\sqrt d}+\\sum_{j=1}^{N} e^{q^\\top k_j/\\sqrt d}}.$$\n\nOrdinarily $\\tilde k_i$ and $\\tilde v_i$ are learned by gradient descent; here they are computed.',
    zh: 'Prefix-tuning 在每个 head 前拼接 $m$ 对额外的 key/value $(\\tilde k_i, \\tilde v_i)$，于是该 head 计算\n\n$$\\operatorname{Attn}(q)=\\frac{\\sum_{i=1}^{m} e^{q^\\top \\tilde k_i/\\sqrt d}\\,\\tilde v_i+\\sum_{j=1}^{N} e^{q^\\top k_j/\\sqrt d}\\,v_j}{\\sum_{i=1}^{m} e^{q^\\top \\tilde k_i/\\sqrt d}+\\sum_{j=1}^{N} e^{q^\\top k_j/\\sqrt d}}.$$\n\n通常 $\\tilde k_i$ 与 $\\tilde v_i$ 由梯度下降学习得到；在这里，它们是直接算出来的。',
  },
  'm.thm.h': {
    en: 'Theorem (informal)',
    zh: '定理（非正式表述）',
  },
  'm.thm': {
    en: 'Let $x\'=(x\'_1,\\dots,x\'_p)$ be a source input whose greedy continuation under $T$ is $y=(y_1,\\dots,y_k)$, and let $x=(x_1,\\dots,x_n)$ be any input with $n=p$ and $x_n=x\'_p$. Assume a mild non-degeneracy condition on the attention logits at every head (condition (10) in the paper, never violated in our experiments). Then there is a prefix $P$ such that the prefixed model $T_P$, decoding greedily from $x$, outputs exactly $y$. The two conditions on $x$ are needed because $T$ has residual connections and positional encodings; the paper shows that neither can be dropped, and that the source input cannot be dispensed with. The proof constructs $P$ by solving one small linear system per head.',
    zh: '设 source $x\'=(x\'_1,\\dots,x\'_p)$ 在模型 $T$ 下的贪心续写为 $y=(y_1,\\dots,y_k)$，$x=(x_1,\\dots,x_n)$ 为任意满足 $n=p$ 且 $x_n=x\'_p$ 的输入，并假设每个 head 的 attention logit 满足一个温和的非退化条件（论文中的条件 (10)，实验中从未被违反）。则存在 prefix $P$，使接入 prefix 的模型 $T_P$ 从 $x$ 出发贪心解码时恰好输出 $y$。之所以需要对 $x$ 的两个条件，是因为 $T$ 含有残差连接和位置编码；论文证明这两个条件都不能去掉，source 的存在也不能省略。证明通过在每个 head 上求解一个小型线性方程组来构造 $P$。',
  },
  'm.am.h': {
    en: 'Attention matching',
    zh: 'Attention Matching',
  },
  'm.am1': {
    en: 'Run $T$ on $x\'$ followed by $y$, and let $q_t$ be the query at the position that produces $y_{t+1}$, for $t=0,\\dots,k-1$. In this source run, $q_t$ attends to its $p+t$ visible tokens with weights $R_{t,j}$, and the head outputs the vector\n\n$$a_t=\\sum_{j} R_{t,j}\\,v_j\\;\\in\\mathbb{R}^{d}.$$\n\nIf every head in every layer outputs the exact same values at these positions during a new input, the hidden states will match the source run, and identical inputs are fed into the residual streams and MLPs. As a result, the model naturally generates $y$ all over again.',
    zh: '让 $T$ 在 $x\'$ 后接 $y$ 上运行一次，记 $q_t$ 为生成 $y_{t+1}$ 的位置上的 query，$t=0,\\dots,k-1$。在这次 source 运行中，$q_t$ 以权重 $R_{t,j}$ 关注它可见的 $p+t$ 个 token，该 head 输出向量\n\n$$a_t=\\sum_{j} R_{t,j}\\,v_j\\;\\in\\mathbb{R}^{d}.$$\n\n如果在新输入上，每一层的每个 head 在这些位置都输出完全相同的值，隐藏状态就会与 source 运行一致，并且残差流和 MLP 接收的输入完全相同。因此，模型自然会再次生成 $y$。',
  },
  'm.am2': {
    en: 'To achieve this, give each query its own prefix slot. Let $S_{t,i}$ be the softmax weight of $q_t$ on slot $i$, computed among the prefix slots only. If the prefix captures almost all of the attention, the head outputs $\\tilde a_t=\\sum_i S_{t,i}\\,\\tilde v_i$, and we want $\\tilde a_t=a_t$. [The figure](#fig-mech) contrasts the two attention patterns for a single query.',
    zh: '为此，为每个 query 分配一个专属的 prefix slot。记 $S_{t,i}$ 为 $q_t$ 在 prefix slot $i$ 上的 softmax 权重（只在 prefix slot 之间归一化）。如果 prefix 吸收了几乎全部注意力，该 head 的输出就是 $\\tilde a_t=\\sum_i S_{t,i}\\,\\tilde v_i$，而我们希望 $\\tilde a_t=a_t$。[下图](#fig-mech)对比了同一个 query 的两种注意力分布。',
  },
  'm.am3': {
    en: 'Writing all $k$ conditions at once gives a linear system for the value prefix:\n\n$$S\\,\\tilde V = R\\,V_{\\mathrm{src}} \\quad\\Longleftrightarrow\\quad \\tilde V = W\\,V_{\\mathrm{src}},\\qquad W=S^{-1}R,$$\n\nwhere row $t$ of $R$ is the source attention of $q_t$ and $V_{\\mathrm{src}}$ stacks the source value vectors. The implementation solves $SW=R$ in float64. Because each slot key is aimed at its own query, $S$ is close to the identity and the solve is well conditioned. [The figure](#fig-svr) shows the three matrices for one head of the [demo above](#demo).',
    zh: '把全部 $k$ 个条件写在一起，就得到关于 value prefix 的线性方程组：\n\n$$S\\,\\tilde V = R\\,V_{\\mathrm{src}} \\quad\\Longleftrightarrow\\quad \\tilde V = W\\,V_{\\mathrm{src}},\\qquad W=S^{-1}R,$$\n\n 其中 $R$ 的第 $t$ 行是 $q_t$ 在 source 运行中的注意力分布，$V_{\\mathrm{src}}$ 由 source 的 value 向量堆叠而成。实现中以 float64 精度求解 $SW=R$。由于每个 slot 的 key 都对准各自的 query，$S$ 接近单位阵，求解的条件数很好。[下图](#fig-svr)给出[上方演示](#demo)中某个 head 的这三个矩阵。',
  },
  'm.qk.h': {
    en: 'Keys under QK-Norm',
    zh: 'QK-Norm 下的 key',
  },
  'm.qk': {
    en: 'Each slot key should make its logit $q_t^\\top \\tilde k/\\sqrt d$ as large as possible. In GPT-2 the logit grows without bound as the key is scaled, so the key is scaled until the logit reaches a chosen value $\\lambda$ ($\\lambda=20$ in the paper). Qwen3 normalizes keys: every key has the form $\\tilde k = u\\odot\\gamma$ with $\\|u\\|=\\sqrt d$, where $\\gamma$ is the key-norm weight, so scaling has no effect. By Cauchy–Schwarz,\n\n$$\\frac{q^\\top (u\\odot\\gamma)}{\\sqrt d}=\\frac{(q\\odot\\gamma)^\\top u}{\\sqrt d}\\;\\le\\;\\|q\\odot\\gamma\\|,\\qquad\\text{with equality at } u^\\star=\\sqrt d\\,\\frac{q\\odot\\gamma}{\\|q\\odot\\gamma\\|}.$$\n\nThe construction therefore uses $\\tilde k = u^\\star\\odot\\gamma$, and the attainable logit $\\|q\\odot\\gamma\\|$ takes the place of $\\lambda$.',
    zh: '每个 slot 的 key 都应使其 logit $q_t^\\top \\tilde k/\\sqrt d$ 尽可能大。在 GPT-2 中，放大 key 可以让 logit 无限增大，因此把 key 放大到 logit 等于设定值 $\\lambda$ 为止（论文取 $\\lambda=20$）。Qwen3 对 key 做了归一化：每个 key 都形如 $\\tilde k = u\\odot\\gamma$，其中 $\\|u\\|=\\sqrt d$，$\\gamma$ 为 key 归一化层的权重，因此放大不起作用。由 Cauchy–Schwarz 不等式，\n\n$$\\frac{q^\\top (u\\odot\\gamma)}{\\sqrt d}=\\frac{(q\\odot\\gamma)^\\top u}{\\sqrt d}\\;\\le\\;\\|q\\odot\\gamma\\|,\\qquad\\text{当 } u^\\star=\\sqrt d\\,\\frac{q\\odot\\gamma}{\\|q\\odot\\gamma\\|} \\text{ 时取等号}.$$\n\n因此构造中取 $\\tilde k = u^\\star\\odot\\gamma$，可达到的 logit $\\|q\\odot\\gamma\\|$ 取代了 $\\lambda$。',
  },
  'm.gqa.h': {
    en: 'Grouped-query attention and repetition',
    zh: 'Grouped-query attention 与重复',
  },
  'm.gqa': {
    en: 'In Qwen3-4B, $G=4$ query heads share each KV head, so a single value prefix must serve all four. Stacking their systems gives\n\n$$\\begin{bmatrix} S^{(1)}\\\\ \\vdots\\\\ S^{(G)} \\end{bmatrix} W=\\begin{bmatrix} R^{(1)}\\\\ \\vdots\\\\ R^{(G)} \\end{bmatrix},$$\n\na system of $Gk$ equations in the $m$ slots of the KV head, which is square when $m=Gk$. Finally, to make “the prefix captures almost all of the attention” ([Attention matching](#attention-matching)) hold, every slot is repeated $r$ times. Inside the prefix the softmax ratios, and therefore $S$ and the solution, are unchanged, but the share of attention that goes to the prefix grows:\n\n$$\\frac{r\\sum_i e^{\\ell_i}}{r\\sum_i e^{\\ell_i}+\\sum_j e^{\\ell\'_j}}\\;\\xrightarrow[\\;r\\to\\infty\\;]{}\\;1,$$\n\nwhere $\\ell_i$ are the prefix logits and $\\ell\'_j$ those of the real tokens.',
    zh: '在 Qwen3-4B 中，每个 KV head 被 $G=4$ 个 query head 共享，因此一组 value prefix 必须同时服务这四个 head。把它们的方程组堆叠起来，得到\n\n$$\\begin{bmatrix} S^{(1)}\\\\ \\vdots\\\\ S^{(G)} \\end{bmatrix} W=\\begin{bmatrix} R^{(1)}\\\\ \\vdots\\\\ R^{(G)} \\end{bmatrix},$$\n\n这是关于该 KV head 的 $m$ 个 slot 的 $Gk$ 个方程，当 $m=Gk$ 时为方阵。最后，为了使“prefix 吸收了几乎全部注意力”（见 [Attention Matching](#attention-matching)）成立，每个 slot 重复 $r$ 次。prefix 内部的 softmax 比例不变，因此 $S$ 和解都不变，但 prefix 获得的注意力份额会增大：\n\n$$\\frac{r\\sum_i e^{\\ell_i}}{r\\sum_i e^{\\ell_i}+\\sum_j e^{\\ell\'_j}}\\;\\xrightarrow[\\;r\\to\\infty\\;]{}\\;1,$$\n\n其中 $\\ell_i$ 为 prefix 的 logit，$\\ell\'_j$ 为真实 token 的 logit。',
  },
  'm.watch.h': {
    en: 'Watching it work',
    zh: '观察实际运行',
  },
  'm.watch': {
    en: 'The view below checks the matching on the numbers of the last message in the [chat](#demo). Choose a layer, a query head and the output word being written. The top row is the source run: $q_t$ spreads its attention over the source with weights $R_{t,:}$ and outputs $a_t$. The middle row is your message with the prefix: the same query sends a share $\\pi_t$ ([defined above](#repeats)) of its attention to the prefix, almost all of it to its own slot, and the rest to your words, from which it reads some vector $b_t$. The head therefore outputs\n\n$$\\tilde a_t=\\pi_t\\,S_{t,:}\\tilde V+(1-\\pi_t)\\,b_t=\\pi_t\\,a_t+(1-\\pi_t)\\,b_t .$$\n\nThe two attention patterns look nothing alike, yet the outputs agree. The bottom strip is the difference $\\tilde a_t-a_t=(1-\\pi_t)(b_t-a_t)$, which grows as you lower $r$ and $\\pi_t$ falls. Without the prefix only $b_t$ is left. Each strip shows the $d=128$ coordinates of the vector, blue for negative and red for positive; bar heights are attention weights on a common 0–100% scale.',
    zh: '下图用[上方聊天](#demo)中最后一条消息的实际数值来检验这种匹配。可以选择层、query head，以及正在写出的输出词。上面一行是 source 运行：$q_t$ 以权重 $R_{t,:}$ 把注意力分散在 source 上，输出 $a_t$。中间一行是带 prefix 的你的消息：同一个 query 把份额为 $\\pi_t$（[定义见上文](#repeats)）的注意力放在 prefix 上（几乎全部落在它自己的 slot），其余落在你的消息上，并从中读出某个向量 $b_t$。于是该 head 的输出为\n\n$$\\tilde a_t=\\pi_t\\,S_{t,:}\\tilde V+(1-\\pi_t)\\,b_t=\\pi_t\\,a_t+(1-\\pi_t)\\,b_t .$$\n\n 两种注意力分布截然不同，输出却一致。最下面一条是两者之差 $\\tilde a_t-a_t=(1-\\pi_t)(b_t-a_t)$：减小 $r$ 会使 $\\pi_t$ 下降，差值随之变大；去掉 prefix，就只剩下 $b_t$。每一条色带表示向量的 $d=128$ 个分量，蓝色为负、红色为正；柱高是注意力权重，统一按 0–100% 的刻度绘制。',
  },
  'mt.step': { en: 'Output word being written:', zh: '正在写出的词：' },
  'mt.vecKey': {
    en: 'The output of a head is a vector, not a number: one cell per coordinate, $d=128$ in all; blue is negative, red positive.',
    zh: '一个 head 的输出是向量而不是一个数：每一格是一个分量，共 $d=128$ 个；蓝色为负，红色为正。',
  },
  'mt.src': {
    en: 'Source run · $q_t$ attends to the source with weights $R_{t,:}$',
    zh: 'source 运行 · $q_t$ 以权重 $R_{t,:}$ 关注 source',
  },
  'mt.on': {
    en: 'Your message with the prefix · the same $q_t$, now almost entirely on the prefix',
    zh: '你的消息 + prefix · 同一个 $q_t$，注意力几乎全部落在 prefix 上',
  },
  'mt.off': {
    en: 'Your message without the prefix · the query can only attend to your words',
    zh: '你的消息，无 prefix · query 只能关注你的消息',
  },
  'mt.g.hidden': { en: 'hidden prompt', zh: '隐藏 prompt' },
  'mt.g.source': { en: 'source', zh: 'source' },
  'mt.g.written': { en: 'written so far', zh: '已写出的词' },
  'mt.g.prefix': { en: 'prefix: {m} slots × {r} copies', zh: 'prefix：{m} 个 slot × {r} 份' },
  'mt.g.msg': { en: 'your message', zh: '你的消息' },
  'mt.srcUnknown': {
    en: 'This prefix was uploaded, so its source run is not available here. The output $a_t$ is recovered from the prefix itself as $S_{t,:}\\tilde V$.',
    zh: '这个 prefix 是上传的，因此这里无法显示它的 source 运行。输出 $a_t$ 由 prefix 本身按 $S_{t,:}\\tilde V$ 还原。',
  },
  'mt.same': {
    en: 'Relative difference {rel}: the same output, so the next layer receives what it received in the source run.',
    zh: '相对差异 {rel}：输出相同，下一层收到的与 source 运行时一样。',
  },
  'mt.close': {
    en: 'Relative difference {rel}: small. Such gaps rarely change which token wins, which is why the paper already reports 99.65% success at $r=1$; raising $r$ closes them.',
    zh: '相对差异 {rel}：很小。这样的差距很少改变哪个 token 胜出，所以论文在 $r=1$ 时就已有 99.65% 的成功率；增大 $r$ 会把它进一步缩小。',
  },
  'mt.differ': {
    en: 'Relative difference {rel}: a clearly different output. Errors of this size, added up over layers, push the model off $y$.',
    zh: '相对差异 {rel}：输出明显不同。这样大小的误差在各层累积，会让模型偏离 $y$。',
  },
  'mt.capOn': {
    en: 'Writing “{w}”, layer {l}, query head {h}. In the source run the query attends only to the source; its largest weight is {top}. With the prefix, a share $\\pi_t$ = {pi} of the attention goes to the prefix and {own} of that to the query’s own slot, so the output matches $a_t$ to within {rel}. Lower $r$ to watch the difference grow.',
    zh: '正在写出“{w}”，第 {l} 层，第 {h} 个 query head。source 运行中 query 只关注 source，最大的权重为 {top}。接入 prefix 后，份额 $\\pi_t$ = {pi} 的注意力落在 prefix 上，其中 {own} 落在该 query 自己的 slot，因此输出与 $a_t$ 的差异只有 {rel}。减小 $r$ 可以看到差异变大。',
  },
  'mt.capOff': {
    en: 'Writing “{w}”, layer {l}, query head {h}. Without the prefix the query can only look at your message, so its output $b_t$ differs from $a_t$ by {rel}, and the model writes its own reply.',
    zh: '正在写出“{w}”，第 {l} 层，第 {h} 个 query head。没有 prefix 时，query 只能关注你的消息，其输出 $b_t$ 与 $a_t$ 相差 {rel}，模型于是写出它自己的回复。',
  },
  'm.watch.empty': {
    en: 'Send a message in the chat above to see this view.',
    zh: '在上方聊天框发送一条消息后，这里会显示对应的注意力图。',
  },
  'res.cond': {
    en: 'Conditioning ([the S, W, R figure](#fig-svr)) explains much of the difference between the two models. On GPT-2, we attribute the decline with output length to ill-conditioning of $S$. On Qwen3-4B, QK-Norm bounds every logit; we conjecture that this keeps $S$ well conditioned and makes repetition rarely necessary.',
    zh: '条件数（见 [S、W、R 图](#fig-svr)）在很大程度上解释了两个模型之间的差异。在 GPT-2 上，我们将成功率随输出长度下降归因于 $S$ 的病态条件。在 Qwen3-4B 上，QK-Norm 使每个 logit 有界；我们推测这使 $S$ 保持良好的条件数，因而很少需要重复。',
  },
  'lim.1': {
    en: "The input must end with the source's last token, $x_n=x'_p$. The [demo](#demo) makes both end with a line break ([why](#why-source)).",
    zh: "输入必须与 source 以同一个 token 结尾，即 $x_n=x'_p$。[演示](#demo)中两者都以换行结尾（[原因](#why-source)）。",
  },
  'lim.2': {
    en: '[The theorem](#theorem) assumes $n=p$: the prefix is built for the positions of the source run. Qwen3 encodes positions with RoPE, which rotates every query and key by an angle proportional to its position, so a logit depends only on the distance between the two. Before each run, the [demo](#demo) therefore rotates the prefix keys by the angle of $n-p$ positions. Each query then sees the prefix at the same relative position as the source query did at the same step, and attention among the input\'s own tokens is unchanged. The prefix file stays the same; only its placement relative to the input changes. GPT-2 has learned absolute positions, so there the rotation is not available, but length differences are tolerated: with a 50-token source, user inputs of 5 to 50 tokens all succeed and 100-token inputs succeed 92% of the time (paper, Appendix C).',
    zh: '[定理](#theorem)假设 $n=p$：prefix 是按 source 运行时的位置构造的。Qwen3 用 RoPE 编码位置，它把每个 query 和 key 旋转一个与其位置成正比的角度，因此 logit 只取决于两者之间的距离。所以[演示](#demo)在每次运行前把 prefix 的 key 旋转 $n-p$ 个位置对应的角度。这样，每个 query 看到 prefix 时的相对位置与 source 运行中同一步的 query 相同，而输入自身 token 之间的 attention 不变。prefix 文件本身不变，变的只是它相对于输入的位置。GPT-2 使用学习得到的绝对位置编码，无法这样旋转，但它能容忍长度差异：source 为 50 个 token 时，5 到 50 个 token 的用户输入全部成功，100 个 token 的输入成功率为 92%（论文附录 C）。',
  },
  'lim.3': {
    en: 'Each KV head needs $m=Gk$ slots. When $Gk$ exceeds the head dimension $d$, $S$ becomes badly conditioned, so in practice $Gk\\lesssim d$; on Qwen3-4B ($G=4$, $d=128$) this suggests roughly $32$ output tokens. We have measured up to $k=10$.',
    zh: '每个 KV head 需要 $m=Gk$ 个 slot。当 $Gk$ 超过 head 维度 $d$ 时，$S$ 会变得严重病态，因此实践中需要 $Gk\\lesssim d$；在 Qwen3-4B（$G=4$，$d=128$）上，这意味着大约 $32$ 个输出 token。我们实测到 $k=10$。',
  },
  'lim.5': {
    en: 'One prefix serves one input–output pair. With two or more pairs the problem is out of reach: the paper gives a two-element dataset that even full fine-tuning cannot realize (Theorem 2.2).',
    zh: '一个 prefix 只对应一对输入输出。两对及以上时问题就超出了能力范围：论文给出了一个只有两个元素、连完整 fine-tuning 都无法实现的数据集（Theorem 2.2）。',
  },
} satisfies Record<string, Entry>;

export type Key = keyof typeof D;

let lang: Lang = pickInitial();

function pickInitial(): Lang {
  try {
    const saved = localStorage.getItem('am-lang');
    if (saved === 'en' || saved === 'zh') return saved;
  } catch {
    /* storage blocked */
  }
  return navigator.language?.toLowerCase().startsWith('zh') ? 'zh' : 'en';
}

export const getLang = () => lang;

export function setLang(l: Lang) {
  lang = l;
  try {
    localStorage.setItem('am-lang', l);
  } catch {
    /* storage blocked */
  }
  applyStatic();
}

export function fill(s: string, p?: Params) {
  return p ? s.replace(/\{(\w+)\}/g, (_, k: string) => String(p[k] ?? `{${k}}`)) : s;
}

export function t(key: Key, p?: Params): string {
  return fill(D[key][lang], p);
}

/** Pick between an inline pair, for text that lives next to its code (algorithm steps). */
export const tr = (en: string, zh: string, p?: Params) => fill(lang === 'zh' ? zh : en, p);

export function applyStatic() {
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  document.title = t('doc.title');
  document.querySelectorAll<HTMLElement>('[data-i18n]').forEach((el) => {
    const v = t(el.dataset.i18n as Key);
    if (el.tagName === 'DIV') {
      // a block: blank lines separate paragraphs; a paragraph that is only $$…$$ is an equation
      el.innerHTML = v
        .split('\n\n')
        .map((p) => (/^\$\$[\s\S]*\$\$$/.test(p.trim()) ? `<div class="eq">${rich(p.trim())}</div>` : `<p>${rich(p)}</p>`))
        .join('');
    } else if (hasMath(v)) el.innerHTML = rich(v);
    else el.textContent = v;
  });
  document.querySelectorAll<HTMLElement>('[data-i18n-html]').forEach((el) => {
    el.innerHTML = rich(t(el.dataset.i18nHtml as Key), true);
  });
  document.querySelectorAll<HTMLElement>('[data-i18n-aria]').forEach((el) => {
    el.setAttribute('aria-label', t(el.dataset.i18nAria as Key));
  });
}
