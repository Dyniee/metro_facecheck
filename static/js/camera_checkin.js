// 1. LẤY CÁC ELEMENT (Đã cập nhật đúng ID cho checkin.html mới)
const video = document.getElementById('video');
const startBtn = document.getElementById('start');
const stationInput = document.getElementById('station');
const statusBox = document.getElementById('status');
const statusIcon = statusBox.querySelector('.status-icon');
const statusTitle = document.getElementById('status-title');
const statusSubtitle = document.getElementById('status-subtitle');
const cameraOverlay = document.getElementById('camera-overlay');

const CHECKIN_INTERVAL = 3000; // Đặt lại 3 giây
let isChecking = false;

// 2. LOGIC KÍCH HOẠT NÚT (Sửa lỗi nút không bấm được)
stationInput.addEventListener('change', () => {
    if (stationInput.value) {
        startBtn.disabled = false;
        // ✨ LOGIC MỈM CƯỜI: Cập nhật hướng dẫn ✨
        setStatus('status-idle', 'Sẵn sàng', 'Nhấn "Bắt đầu Check-in" để mở camera.');
    } else {
        startBtn.disabled = true;
        setStatus('status-idle', 'Chưa Sẵn Sàng', 'Vui lòng chọn ga của bạn để bắt đầu.');
    }
});

// 3. HÀM SET TRẠNG THÁI (Dùng cho giao diện mới)
const setStatus = (state, title, subtitle) => {
    statusBox.className = 'status-box ' + state;
    statusTitle.innerText = title;
    statusSubtitle.innerText = subtitle;
    if (state === 'status-checking') {
        statusIcon.innerHTML = '<div class="icon-spinner"></div>';
        cameraOverlay.classList.add('visible');
    } else if (state === 'status-pass') {
        statusIcon.innerHTML = '<svg><use href="#icon-pass"></use></svg>';
        cameraOverlay.classList.remove('visible');
    } else if (state === 'status-deny') {
        statusIcon.innerHTML = '<svg><use href="#icon-deny"></use></svg>';
        cameraOverlay.classList.remove('visible');
    } else if (state === 'status-error') {
        statusIcon.innerHTML = '<svg><use href="#icon-error"></use></svg>';
        cameraOverlay.classList.remove('visible');
    } else {
        statusIcon.innerHTML = '<svg><use href="#icon-idle"></use></svg>';
        cameraOverlay.classList.remove('visible');
    }
};

// 4. HÀM CHECK-IN CHÍNH
const performCheck = async () => {
    if (isChecking) return;
    isChecking = true;

    const currentStation = stationInput.value;
    if (!currentStation) {
        setStatus('status-error', 'ĐÃ DỪNG', 'Vui lòng chọn một ga để tiếp tục quét.');
        isChecking = false;
        setTimeout(performCheck, CHECKIN_INTERVAL);
        return;
    }

    setStatus('status-checking', 'ĐANG QUÉT', 'Giữ yên, mở mắt và nhìn thẳng camera...');
    video.classList.add('flash');

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    
    // Lật ảnh (mirror)
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const data = canvas.toDataURL('image/jpeg');

    setTimeout(() => video.classList.remove('flash'), 300);

    try {
        const res = await fetch('http://127.0.0.1:5000/api/checkin', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_b64: data,
                station: currentStation // Gửi TÊN GA
            })
        });

        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const j = await res.json();

        if (j.success) {
            setStatus('status-pass', 'THÀNH CÔNG', j.message || `User ${j.user_id} | Vé: ${j.reason || 'Hợp lệ'}`);
        } else {
            // Tự động hiển thị lỗi "Không phát hiện nụ cười" (j.message) từ app.py
            setStatus('status-deny', 'TỪ CHỐI', j.message || `Lý do: ${j.reason || 'Không nhận diện được'}`);
        }

    } catch (e) {
        console.error('Lỗi Check-in:', e);
        setStatus('status-error', 'LỖI', 'Không thể kết nối máy chủ. Đang thử lại...');
    }

    setTimeout(() => {
        isChecking = false;
        performCheck();
    }, CHECKIN_INTERVAL);
};

// 5. NÚT START
startBtn.addEventListener('click', async () => {
    startBtn.disabled = true;
    startBtn.innerHTML = '<div class="icon-spinner"></div> Đang khởi động...';

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();

        video.oncanplay = () => {
            startBtn.innerHTML = 'ĐANG CHẠY...';
            startBtn.style.opacity = 0.7;
            performCheck();
        };

    } catch (err) {
        console.error('Lỗi Camera:', err);
        setStatus('status-error', 'LỖI CAMERA', 'Không thể mở camera. Vui lòng cấp quyền.');
        startBtn.disabled = false;
        startBtn.innerText = 'Thử lại';
    }
});