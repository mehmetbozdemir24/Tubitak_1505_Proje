// Config
const API_URL = "http://localhost:8000";

// State
let currentState = {
    user: null, // { role: 'admin' | 'manager' | 'user', name: '...' }
    route: 'auth' // 'auth' | 'chat' | 'upload' | 'history'
};

// DOM Elements
const app = document.getElementById('app');

// --- ROUTING & RENDERING ---

function init() {
    render();
    window.addEventListener('hashchange', handleHashChange);
}

function handleHashChange() {
    const hash = window.location.hash.slice(1);
    if (!currentState.user && hash !== 'auth') {
        window.location.hash = 'auth';
        return;
    }
    currentState.route = hash || 'auth';
    render();
}

function navigate(route) {
    window.location.hash = route;
}

function render() {
    app.innerHTML = '';

    if (!currentState.user) {
        renderAuth();
    } else {
        renderLayout();
    }
}

// --- VIEWS ---

function renderAuth() {
    const container = document.createElement('div');
    container.className = 'auth-container';
    container.innerHTML = `
        <div class="auth-card">
            <h1 style="color:var(--primary-color); margin-bottom:10px;">Teracity</h1>
            <h3 style="margin-bottom:30px; font-weight:normal;">TÜBİTAK 1505 Asistanı</h3>
            <p style="margin-bottom:20px; color:#777;">Giriş yapmak için rolünüzü seçin</p>
            
            <div style="margin-bottom: 20px; text-align: left;">
                <label for="role-select" style="display:block; margin-bottom:5px; color:#555; font-size:0.9rem;">Kullanıcı Rolü</label>
                <select id="role-select" class="role-select">
                    <option value="user">USER (Personel)</option>
                    <option value="editor">EDITOR (Düzenleyici)</option>
                    <option value="manager">MANAGER (Yönetici)</option>
                    <option value="admin">ADMIN (Sistem Yöneticisi)</option>
                </select>
            </div>

            <button class="login-submit-btn" onclick="handleLoginSubmit()">
                Giriş Yap <i class="fas fa-arrow-right" style="margin-left:10px;"></i>
            </button>
        </div>
    `;
    app.appendChild(container);
}

function handleLoginSubmit() {
    const role = document.getElementById('role-select').value;
    login(role);
}

function renderLayout() {
    const layout = document.createElement('div');
    layout.style.height = '100%';
    layout.style.display = 'flex';
    layout.style.flexDirection = 'column';

    // Header
    layout.innerHTML = `
        <header>
            <a href="#" class="logo">
                <i class="fas fa-robot"></i> Teracity AI
            </a>
            <div class="user-info">
                <span><i class="fas fa-circle" style="color:SpringGreen; font-size:10px;"></i> ${currentState.user.role.toUpperCase()}</span>
                <button class="logout-btn" onclick="logout()">Çıkış</button>
            </div>
        </header>
        <div class="dashboard">
            <aside class="sidebar">
                <nav>
                    <a href="#chat" class="nav-item ${currentState.route === 'chat' ? 'active' : ''}">
                        <i class="fas fa-comment-dots"></i> Sohbet
                    </a>
                    ${currentState.user.role !== 'user' ? `
                    <a href="#upload" class="nav-item ${currentState.route === 'upload' ? 'active' : ''}">
                        <i class="fas fa-cloud-upload-alt"></i> Belge Yükle
                    </a>
                    ` : ''}
                </nav>
            </aside>
            <main class="content-area" id="main-content"></main>
        </div>
    `;

    app.appendChild(layout);

    // Render Child Views
    const mainContent = document.getElementById('main-content');
    if (currentState.route === 'chat') renderChat(mainContent);
    else if (currentState.route === 'upload') renderUpload(mainContent);
}

function renderChat(container) {
    container.innerHTML = `
        <div class="chat-window" id="chat-window">
            <div class="message bot">
                Merhaba! Ben Teracity yapay zeka asistanı. Size nasıl yardımcı olabilirim?
                <div class="message-meta">Az önce</div>
            </div>
        </div>
        <div class="chat-input-area">
            <input type="text" id="user-input" placeholder="Bir soru sorun..." onkeypress="handleEnter(event)">
            <button class="send-btn" onclick="sendMessage()"><i class="fas fa-paper-plane"></i></button>
        </div>
    `;
}

function renderUpload(container) {
    container.innerHTML = `
        <div class="upload-container">
            <h2>Yeni Belge Ekle</h2>
            <p style="color:#666; margin-bottom:30px;">Sisteme yeni PDF veya Word dokümanları yükleyin.</p>
            
            <div class="drop-zone" onclick="document.getElementById('file-input').click()">
                <i class="fas fa-cloud-upload-alt" style="font-size:3rem; color:var(--primary-color); margin-bottom:20px;"></i>
                <p>Dosyaları buraya sürükleyin veya seçmek için tıklayın</p>
                <input type="file" id="file-input" hidden multiple onchange="handleFileSelect(event)">
            </div>
            
            <div id="file-list" style="margin-top:20px; text-align:left;"></div>
        </div>
    `;
}

// --- LOGIC ---

function login(role) {
    let name = 'Kullanıcı';
    if (role === 'admin') name = 'Sistem Yöneticisi';
    else if (role === 'manager') name = 'Yönetici';
    else if (role === 'editor') name = 'Editör';

    currentState.user = {
        role: role,
        name: name
    };
    navigate('chat');
}

function logout() {
    currentState.user = null;
    navigate('auth');
}

function handleEnter(e) {
    if (e.key === 'Enter') sendMessage();
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    if (!message) return;

    // Add User Message
    addMessage(message, 'user');
    input.value = '';

    // Show Loading
    const loadingId = addMessage('Thinking...', 'bot', true);

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: message,
                role: currentState.user.role
            })
        });

        const data = await response.json();

        // Remove Loading and Show Response
        document.getElementById(loadingId).remove();

        let responseContent = data.answer;
        if (data.context_used && data.context_used.length > 0) {
            responseContent += `<br><br><small><strong>Kaynaklar:</strong></small>`;
            // Basitçe kaynakları gösterelim
            responseContent += `<div style="margin-top:5px; font-size:0.8rem; background:rgba(0,0,0,0.05); padding:5px;">`;
            // İlk kaynağın bir kısmını göster
            responseContent += `... ${data.context_used[0].substring(0, 100)} ...`;
            responseContent += `</div>`;
        }

        addMessage(responseContent, 'bot');

    } catch (error) {
        console.error(error);
        document.getElementById(loadingId).remove();
        addMessage('Bir hata oluştu. Lütfen bağlantınızı kontrol edin.', 'bot');
    }
}

function addMessage(text, sender, isLoading = false) {
    const chatWindow = document.getElementById('chat-window');
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    div.innerHTML = text;
    if (isLoading) {
        div.id = 'loading-msg';
        div.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Yanıt üretiliyor...';
    }
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return div.id;
}

async function handleFileSelect(event) {
    const files = event.target.files;
    const list = document.getElementById('file-list');
    list.innerHTML = '';

    for (const file of files) {
        // Show uploading state
        const item = document.createElement('div');
        item.style.padding = '10px';
        item.style.borderBottom = '1px solid #eee';
        item.innerHTML = `
            <i class="far fa-file"></i> ${file.name} 
            <span id="status-${file.name.replace(/\s/g, '')}" style="float:right; color:blue;">
                <i class="fas fa-spinner fa-spin"></i> Yükleniyor...
            </span>
        `;
        list.appendChild(item);

        // Upload
        await uploadFile(file);
    }
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('role', currentState.user.role);

    const statusId = `status-${file.name.replace(/\s/g, '')}`;
    const statusEl = document.getElementById(statusId);

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            statusEl.innerHTML = `<i class="fas fa-check-circle"></i> Tamamlandı (${result.chunks_count} parça)`;
            statusEl.style.color = 'green';
        } else {
            const err = await response.json();
            statusEl.innerHTML = `<i class="fas fa-times-circle"></i> Hata: ${err.detail}`;
            statusEl.style.color = 'red';
        }
    } catch (error) {
        console.error(error);
        statusEl.innerHTML = `<i class="fas fa-times-circle"></i> Bağlantı Hatası`;
        statusEl.style.color = 'red';
    }
}

// Start
init();
