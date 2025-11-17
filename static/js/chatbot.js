document.addEventListener('DOMContentLoaded', function() {
    // 1. Lấy các Element
    const chatButton = document.getElementById('chat-widget-btn');
    const chatWindow = document.getElementById('chat-window');
    const closeButton = document.getElementById('chat-close-btn');
    const sendButton = document.getElementById('chat-send-btn');
    const chatInput = document.getElementById('chat-input');
    const chatBody = document.getElementById('chat-body');

    // (Nếu không tìm thấy các element, có thể người dùng chưa đăng nhập, thoát)
    if (!chatButton || !chatWindow) {
        return;
    }

    // 2. Gắn sự kiện Mở/Đóng
    chatButton.addEventListener('click', () => {
        chatWindow.classList.toggle('active');
        // Tự động focus vào ô nhập liệu khi mở
        if (chatWindow.classList.contains('active')) {
            chatInput.focus();
        }
    });

    closeButton.addEventListener('click', () => {
        chatWindow.classList.remove('active');
    });

    // 3. Gắn sự kiện Gửi Tin Nhắn
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    /**
     * Hàm chính để gửi tin nhắn
     */
    async function sendMessage() {
        const message = chatInput.value.trim();
        if (message === '') return;

        // 1. Hiển thị tin nhắn của User lên giao diện
        addMessageToUI(message, 'user');
        chatInput.value = ''; // Xóa ô nhập

        // 2. Hiển thị trạng thái "Bot đang gõ..."
        addMessageToUI('...', 'bot-typing');

        try {
            // 3. Gọi API /api/chat
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) {
                throw new Error('Lỗi kết nối máy chủ');
            }

            const data = await response.json();
            
            // 4. Xóa trạng thái "Đang gõ..."
            removeTypingIndicator();
            
            // 5. Hiển thị tin nhắn trả lời của Bot
            addMessageToUI(data.reply, 'bot');

        } catch (error) {
            removeTypingIndicator();
            addMessageToUI('Xin lỗi, tôi gặp lỗi kết nối. Vui lòng thử lại.', 'bot');
        }
    }

    /**
     * Hàm trợ giúp để thêm tin nhắn vào cửa sổ chat
     * @param {string} message - Nội dung tin nhắn
     * @param {string} type - 'user', 'bot', hoặc 'bot-typing'
     */
    function addMessageToUI(message, type) {
        const messageEl = document.createElement('div');
        messageEl.className = 'chat-message ' + type;
        
        // Dùng innerHTML để Bot có thể trả về chữ đậm (<strong>)
        messageEl.innerHTML = message; 
        
        chatBody.appendChild(messageEl);
        // Tự động cuộn xuống tin nhắn mới nhất
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    /**
     * Hàm trợ giúp để xóa tin "Đang gõ..."
     */
    function removeTypingIndicator() {
        const typingEl = chatBody.querySelector('.bot-typing');
        if (typingEl) {
            typingEl.remove();
        }
    }
});