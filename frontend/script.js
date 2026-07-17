// ============================================================
// 医学RAG问答系统 - Agent 工作流全链路可视化
// ============================================================

const API_BASE = 'http://localhost:8103';
let isConnected = false;
let theme = localStorage.getItem('agent-theme') || 'light';

// ===== 初始化 =====
document.addEventListener('DOMContentLoaded', () => {
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
    checkConnection();
    setupTextarea();
});

// ===== 连接检查 =====
async function checkConnection() {
    const badge = document.getElementById('statusBadge');
    try {
        const r = await fetch(`${API_BASE}/api/health`);
        if (r.ok) {
            isConnected = true;
            badge.classList.add('connected');
            badge.querySelector('.status-text').textContent = '已连接';
        } else {
            throw new Error('not ok');
        }
    } catch {
        isConnected = false;
        badge.classList.remove('connected');
        badge.querySelector('.status-text').textContent = '未连接';
    }
}

// ===== 主题切换 =====
function toggleTheme() {
    theme = theme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('agent-theme', theme);
}

// ===== 输入框 =====
function setupTextarea() {
    const ta = document.getElementById('messageInput');
    ta.addEventListener('input', () => {
        ta.style.height = 'auto';
        ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
    });
    ta.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

// ===== 折叠工作流面板 =====
function toggleWorkflowPanel() {
    document.getElementById('workflowPanel').classList.toggle('collapsed');
}

// ===== 示例问题 =====
function askExample(q) {
    document.getElementById('messageInput').value = q;
    document.getElementById('messageInput').style.height = 'auto';
    sendMessage();
}

// ===== 发送消息 =====
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const btn = document.getElementById('sendButton');
    const text = input.value.trim();
    if (!text) return;

    if (!isConnected) {
        addErrorMessage('后端服务未连接', '请先启动 python agent.py');
        return;
    }

    input.disabled = true;
    input.value = '';
    input.style.height = 'auto';
    btn.disabled = true;

    // 隐藏欢迎页
    const welcome = document.getElementById('welcomeMessage');
    if (welcome) welcome.style.display = 'none';

    // 添加用户消息
    addUserMessage(text);

    // 显示加载
    const loadEl = addLoadingMessage();

    // 重置管道动画
    resetPipeline();

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: text })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // 移除加载
        if (loadEl.parentNode) loadEl.remove();

        // 添加 Agent 响应
        addBotResponse(data);

        // 驱动管道动画
        animatePipeline(data.thinking_steps || []);

    } catch (err) {
        if (loadEl.parentNode) loadEl.remove();
        addErrorMessage('请求失败', err.message);
    }

    input.disabled = false;
    btn.disabled = false;
    input.focus();
}

// ===== 添加用户消息 =====
function addUserMessage(text) {
    const container = document.getElementById('chatMessages');
    const now = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

    const div = document.createElement('div');
    div.className = 'message';
    div.innerHTML = `
        <div class="msg-user">
            <div class="msg-bubble">${escapeHtml(text)}</div>
        </div>
        <div class="msg-time">${now}</div>
    `;
    container.appendChild(div);
    scrollToBottom();
}

// ===== 添加加载消息 =====
function addLoadingMessage() {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message';
    div.id = 'loadingMsg';
    div.innerHTML = `
        <div class="loading-msg">
            <div class="loading-spinner"></div>
            <span class="loading-text">Agent 正在思考...</span>
        </div>
    `;
    container.appendChild(div);
    scrollToBottom();
    return div;
}

// ===== 添加错误消息 =====
function addErrorMessage(title, detail) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message';
    div.innerHTML = `
        <div class="error-msg">
            <span class="error-icon">⚠️</span>
            <div class="error-content">
                <div class="error-title">${escapeHtml(title)}</div>
                <div class="error-detail">${escapeHtml(detail)}</div>
            </div>
        </div>
    `;
    container.appendChild(div);
    scrollToBottom();
}

// ===== 添加机器人响应（工作流可视化 + 答案） =====
function addBotResponse(data) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message';

    // 解析 thinking steps
    const steps = parseSteps(data.thinking_steps || []);

    // 生成步骤HTML
    let stepsHtml = '';
    steps.forEach((s, i) => {
        stepsHtml += `
            <div class="ts-step">
                <div class="ts-badge badge-${s.cls}">${s.icon}</div>
                <div class="ts-content">
                    <div class="ts-label">${s.label}</div>
                    <div class="ts-detail">${s.detail}</div>
                    ${s.duration ? `<div class="ts-duration">⏱ ${s.duration}</div>` : ''}
                </div>
            </div>
        `;
    });

    // 标签
    const typeMap = {
        'disease_definition': '疾病定义',
        'diet': '饮食建议',
        'drug': '用药咨询',
        'diagnosis': '鉴别诊断',
        'prevention': '预防措施',
        'clinical': '临床表现',
        'general': '综合'
    };
    const typeLabel = typeMap[data.question_type] || data.question_type || '未知';
    const strategyLabel = data.retrieval_strategy === 'hybrid' ? '混合检索' :
                          data.retrieval_strategy === 'qa_only' ? '仅QA库' :
                          data.retrieval_strategy === 'pdf_only' ? '仅PDF库' : data.retrieval_strategy;
    const elapsed = data.elapsed_seconds ? `${data.elapsed_seconds.toFixed(1)}s` : '';

    // 构建回答
    let answerHtml = `<div class="answer-content">${escapeHtml(data.answer || '（无答案返回）')}</div>`;

    // 如果thinking_steps中包含来源文件名，提取展示
    const sources = extractSources(data.thinking_steps || []);
    if (sources.length > 0) {
        answerHtml += `
            <div class="answer-sources" id="sources_${Date.now()}">
                <div class="sources-header" onclick="toggleSources(this)">
                    <span>📎</span>
                    <span>知识来源 (${sources.length})</span>
                    <span style="margin-left:auto;font-size:10px;color:var(--gray-400);">▼</span>
                </div>
                <div class="sources-body">
                    ${sources.map(s => `
                        <div class="source-item">
                            <span class="source-icon">${s.type === 'qa' ? '📚' : '📖'}</span>
                            <span class="source-text">${escapeHtml(s.name)}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    div.innerHTML = `
        <div class="msg-bot">
            <div class="msg-bubble">
                <!-- 思考步骤 -->
                <div class="thinking-steps" id="steps_${Date.now()}">
                    <div class="thinking-steps-header" onclick="toggleSteps(this)">
                        <div class="ts-header-left">
                            <span class="icon">🧠</span>
                            <span class="title">Agent 思考链路</span>
                        </div>
                        <div class="ts-stats">
                            <span class="ts-stat">📌 ${typeLabel}</span>
                            <span class="ts-stat">📊 ${strategyLabel}</span>
                            <span class="ts-stat">📐 ${data.context_length || 0} 字符</span>
                            ${elapsed ? `<span class="ts-stat">⏱ ${elapsed}</span>` : ''}
                            <span class="ts-toggle">▼</span>
                        </div>
                    </div>
                    <div class="ts-body">
                        ${stepsHtml}
                    </div>
                </div>

                <!-- 答案区域 -->
                <div class="answer-area">
                    <div class="answer-header">
                        <span class="icon">💡</span>
                        <span class="title">最终答案</span>
                        <div class="answer-tags">
                            <span class="answer-tag tag-type">${typeLabel}</span>
                            <span class="answer-tag tag-strategy">${strategyLabel}</span>
                            <span class="answer-tag tag-length">${data.context_length || 0} 字符</span>
                        </div>
                    </div>
                    ${answerHtml}
                </div>
            </div>
        </div>
    `;

    container.appendChild(div);
    scrollToBottom();
}

// ===== 解析思考步骤 =====
function parseSteps(steps) {
    if (!steps || steps.length === 0) {
        return [{
            icon: '❓', cls: 'router', label: '未知', detail: '未获取到思考步骤',
        }];
    }

    return steps.map(step => {
        let s = { icon: '🔍', cls: 'router', label: '路由分析', detail: step, duration: '' };

        if (step.includes('[Router Agent]')) {
            s.icon = '🔍'; s.cls = 'router'; s.label = '路由分析 · 意图识别';
            let detail = step.replace('[Router Agent] ', '');
            // 高亮关键信息
            detail = detail
                .replace(/(类型: \w+)/g, '<span class="hl hl-info">$1</span>')
                .replace(/(策略: \w+)/g, '<span class="hl hl-info">$1</span>');
            s.detail = detail;
        }
        else if (step.includes('[Retrieve Agent]')) {
            let detail = step.replace('[Retrieve Agent] ', '');

            if (step.includes('HybridSearch')) {
                s.icon = '📚'; s.cls = 'qa'; s.label = 'QA 问答对库 · 混合检索';
                detail = detail
                    .replace(/HybridSearch Skill/g, '<span class="hl hl-qa">HybridSearch Skill</span>')
                    .replace(/(\d+) 条文档/g, '<span class="hl hl-qa">$1 条文档</span>');
            }
            else if (step.includes('ParentDocSearch')) {
                s.icon = '📖'; s.cls = 'pdf'; s.label = 'PDF 教材库 · 父子文档检索';
                detail = detail
                    .replace(/ParentDocSearch Skill/g, '<span class="hl hl-pdf">ParentDocSearch Skill</span>')
                    .replace(/(\d+) 条段落/g, '<span class="hl hl-pdf">$1 条段落</span>');
                // 提取PDF文件名用于来源显示
                const pdfMatch = step.match(/(\d+\..*?\.pdf)/g);
                if (pdfMatch) {
                    s.detail = detail;
                } else {
                    s.detail = detail;
                }
            }
            else if (step.includes('上下文聚合')) {
                s.icon = '🔗'; s.cls = 'qa'; s.label = '上下文聚合';
                detail = detail
                    .replace(/(\d+) 字符/g, '<span class="hl hl-info">$1 字符</span>');
            }
            else {
                s.icon = '📡'; s.cls = 'qa'; s.label = '检索阶段';
            }
            s.detail = detail;
        }
        else if (step.includes('[Verify Agent]')) {
            s.icon = '✓'; s.cls = 'verify'; s.label = '质量验证';
            let detail = step.replace('[Verify Agent] ', '');
            if (step.includes('通过')) {
                detail = detail.replace(/(通过)/, '<span class="hl hl-success">✅ $1</span>');
            } else if (step.includes('不足')) {
                detail = detail.replace(/(不足)/, '<span class="hl hl-warn">⚠️ $1</span>');
            }
            s.detail = detail;
        }
        else if (step.includes('[Generate Agent]')) {
            s.icon = '🤖'; s.cls = 'generate'; s.label = '答案生成 · LLM 推理';
            let detail = step.replace('[Generate Agent] ', '');
            detail = detail
                .replace(/AnswerGeneration Skill/g, '<span class="hl hl-llm">AnswerGeneration Skill</span>')
                .replace(/(\d+) 字符/g, '<span class="hl hl-llm">$1 字符</span>');
            s.detail = detail;
        }

        return s;
    });
}

// ===== 提取知识来源 =====
function extractSources(steps) {
    const sources = [];
    steps.forEach(step => {
        // 提取PDF来源
        const pdfMatch = step.match(/(\d+\..*?\.pdf)/g);
        if (pdfMatch) {
            pdfMatch.forEach(pdf => {
                if (!sources.find(s => s.name === pdf)) {
                    sources.push({ type: 'pdf', name: pdf });
                }
            });
        }
        // 检查QA库召回
        if (step.includes('HybridSearch') && step.includes('QA')) {
            if (!sources.find(s => s.name === 'QA问答对数据集')) {
                sources.push({ type: 'qa', name: 'QA问答对数据集 (中医/医疗)' });
            }
        }
    });
    return sources;
}

// ===== 重置管道动画 =====
function resetPipeline() {
    document.querySelectorAll('.pipeline-step').forEach(el => {
        el.classList.remove('active', 'done');
    });
}

// ===== 管道动画（逐步高亮） =====
function animatePipeline(steps) {
    // 管道步骤索引映射
    const stepMap = {
        'router': 0,
        'qa': 1,
        'pdf': 2,
        'verify': 3,
        'generate': 4
    };

    // 找出实际执行了哪些步骤
    const executed = [];
    if (steps.some(s => s.includes('[Router Agent]'))) executed.push('router');
    if (steps.some(s => s.includes('HybridSearch'))) executed.push('qa');
    if (steps.some(s => s.includes('ParentDocSearch'))) executed.push('pdf');
    if (steps.some(s => s.includes('[Verify Agent]'))) executed.push('verify');
    if (steps.some(s => s.includes('[Generate Agent]'))) executed.push('generate');

    // 没有匹配到，使用默认全流程动画
    const order = executed.length > 0 ? executed : ['router', 'qa', 'pdf', 'verify', 'generate'];

    // 逐个点亮
    let delay = 300;
    order.forEach((name, i) => {
        const idx = stepMap[name];
        if (idx === undefined) return;
        setTimeout(() => {
            const el = document.querySelector(`.pipeline-step[data-step="${idx}"]`);
            if (el) {
                // 先 active（闪烁），然后变为 done
                el.classList.add('active');
                setTimeout(() => {
                    el.classList.remove('active');
                    el.classList.add('done');
                }, 600);
            }
        }, delay);
        delay += 500;
    });
}

// ===== 折叠/展开思考步骤 =====
function toggleSteps(header) {
    const el = header.closest('.thinking-steps');
    el.classList.toggle('collapsed');
}

// ===== 折叠/展开知识来源 =====
function toggleSources(header) {
    const el = header.closest('.answer-sources');
    el.classList.toggle('collapsed');
}

// ===== 滚动到底部 =====
function scrollToBottom() {
    const container = document.getElementById('chatMessages');
    setTimeout(() => {
        container.scrollTop = container.scrollHeight;
    }, 60);
}

// ===== HTML 转义 =====
function escapeHtml(str) {
    if (!str) return '';
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

// ===== 定期检查连接 =====
setInterval(checkConnection, 30000);
