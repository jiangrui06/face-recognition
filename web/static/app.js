/**
 * 人脸识别打卡系统 - 前端 JavaScript
 *
 * 处理视频流管理、HTMX 增强、实时事件轮询
 */

document.addEventListener('DOMContentLoaded', function() {
    // 每30秒自动刷新仪表盘记录
    const todayRecords = document.getElementById('today-records');
    if (todayRecords) {
        setInterval(() => {
            fetch('/api/attendance/today')
                .then(r => r.json())
                .then(data => {
                    if (data.records && data.records.length > 0) {
                        todayRecords.innerHTML = data.records.map(r => `
                            <tr>
                                <td>${r.name}</td>
                                <td>${r.time}</td>
                                <td>
                                    <span class="badge bg-success">
                                        ${r.confidence ? r.confidence.toFixed(1) : '-'}
                                    </span>
                                </td>
                            </tr>
                        `).join('');
                    }
                })
                .catch(() => {});
        }, 30000);
    }

    // 提示框初始化
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (el) {
        return new bootstrap.Tooltip(el);
    });

    // 键盘快捷键
    document.addEventListener('keydown', function(e) {
        // Ctrl+1 -> 仪表盘, Ctrl+2 -> 打卡, 等等
        if (e.ctrlKey && !e.shiftKey && !e.altKey) {
            const nav = {
                '1': '/',
                '2': '/attendance',
                '3': '/employees',
                '4': '/reports',
            };
            const url = nav[e.key];
            if (url) {
                e.preventDefault();
                window.location.href = url;
            }
        }
    });

    console.log('人脸识别打卡系统已启动');
});

/**
 * 通过 API 重新训练模型
 */
function retrainModel() {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = '⏳ 训练中...';

    fetch('/api/train', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            btn.disabled = false;
            btn.textContent = '🎯 重新训练';
            if (data.success) {
                showToast('success', '模型训练成功！');
            } else {
                showToast('danger', '训练失败: ' + (data.detail || '未知错误'));
            }
        })
        .catch(e => {
            btn.disabled = false;
            btn.textContent = '🎯 重新训练';
            showToast('danger', '错误: ' + e.message);
        });
}

/**
 * 显示 Bootstrap 通知提示
 */
function showToast(type, message) {
    const container = document.getElementById('toast-container');
    if (!container) {
        const div = document.createElement('div');
        div.id = 'toast-container';
        div.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(div);
    }

    const id = 'toast-' + Date.now();
    const html = `
        <div id="${id}" class="toast align-items-center text-white bg-${type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

    document.getElementById('toast-container').insertAdjacentHTML('beforeend', html);
    const toastEl = document.getElementById(id);
    const toast = new bootstrap.Toast(toastEl, { autohide: true, delay: 4000 });
    toast.show();

    toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
}
