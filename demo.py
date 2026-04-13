import streamlit as st
import numpy as np
import time
import pandas as pd
from pathlib import Path
import sys
import pyaudio
import altair as alt

# Thiết lập đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from scripts.inference import AudioEventDetector

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Emergency Sound Detector",
    page_icon="🚨",
    layout="wide"
)

# Tùy chỉnh CSS để giao diện đẹp hơn
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .detection-card { padding: 10px; border-left: 5px solid #ff4b4b; margin-bottom: 10px; background-color: white; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚨 Emergency Audio Event Detection")
st.markdown("Hệ thống phát hiện âm thanh khẩn cấp thời gian thực sử dụng trí tuệ nhân tạo (AI).")

# --- SIDEBAR CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    model_path = st.text_input("Đường dẫn Model", "models/saved_models/best_model.pth")
    config_path = st.text_input("Đường dẫn Config", "configs/config.yaml")
    device = st.selectbox("Thiết bị xử lý", ["cpu", "cuda", "mps"], index=0)
    st.divider()
    threshold = st.slider("Ngưỡng tin cậy (Confidence)", 0.0, 1.0, 0.8)
    st.info("💡 Mẹo: Tăng ngưỡng nếu hệ thống báo nhầm, giảm nếu hệ thống bỏ lỡ âm thanh.")

# --- KHỞI TẠO DETECTOR ---
@st.cache_resource
def load_detector(m_path, c_path, dev):
    try:
        return AudioEventDetector(
            model_path=m_path,
            config_path=c_path,
            device=dev,
            root_path=PROJECT_ROOT
        )
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None

detector = load_detector(model_path, config_path, device)

# --- BỐ CỤC CHÍNH ---
col_viz, col_log = st.columns([2, 1])

with col_viz:
    st.subheader("📊 Tín hiệu âm thanh (Live)")
    # Placeholder cho biểu đồ sóng âm
    chart_placeholder = st.empty()
    
    st.subheader("🎯 Kết quả nhận diện mới nhất")
    # Placeholder cho metric kết quả
    metric_col1, metric_col2 = st.columns(2)
    res_class = metric_col1.empty()
    res_conf = metric_col2.empty()

with col_log:
    st.subheader("📜 Nhật ký phát hiện")
    log_scroll_area = st.container(height=400)
    log_placeholder = log_scroll_area.empty()

# --- LOGIC XỬ LÝ ÂM THANH ---
if detector:
    # Điều khiển Start/Stop bằng Session State
    if "running" not in st.session_state:
        st.session_state.running = False

    # Khởi tạo danh sách lưu log ở session để thẻ mới đẩy lên đầu
    if "event_logs" not in st.session_state:
        st.session_state.event_logs = []

    # Khôi phục log cũ nếu có trên giao diện
    if st.session_state.event_logs:
        with log_placeholder.container():
            st.markdown("".join(st.session_state.event_logs), unsafe_allow_html=True)

    def toggle_run():
        st.session_state.running = not st.session_state.running

    st.button("Bắt đầu / Dừng nghe", on_click=toggle_run, type="primary" if not st.session_state.running else "secondary")

    if st.session_state.running:
        # Thiết lập thông số Audio
        SAMPLING_RATE = 22050
        CHUNK_SIZE = 4410  # 0.2s mỗi lần đọc để biểu đồ mượt hơn
        BUFFER_DURATION = 4  # 4 giây để model xử lý
        MAX_BUFFER = SAMPLING_RATE * BUFFER_DURATION
        
        audio_p = pyaudio.PyAudio()
        stream = audio_p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLING_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        audio_buffer = np.zeros(MAX_BUFFER, dtype=np.float32)
        
        try:
            while st.session_state.running:
                # Đọc dữ liệu từ micro
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.float32)
                
                # Cập nhật buffer (cuộn dữ liệu)
                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk
                # Vẽ biểu đồ sóng âm (Giữ cứng Y axis từ -0.2 đến 0.2 bằng Altair)
                plot_data = pd.DataFrame(audio_buffer[::10], columns=["Amplitude"]).reset_index()
                
                chart = alt.Chart(plot_data).mark_line(color="#2e7bcf", strokeWidth=1.0).encode(
                    x=alt.X('index', axis=alt.Axis(labels=False, ticks=False, title=None)),
                    y=alt.Y('Amplitude', scale=alt.Scale(domain=[-0.2, 0.2], clamp=True), axis=alt.Axis(title=None))
                ).properties(height=150)
                chart_placeholder.altair_chart(chart, width="stretch")
                
                # Chạy AI inference
                result = detector.predict_real_time(audio_buffer, SAMPLING_RATE)
                
                if result and result['confidence'] >= threshold:
                    name = result['class'].upper()
                    conf = result['confidence']
                    curr_time = time.strftime("%H:%M:%S")
                    
                    # Cập nhật Metric
                    res_class.metric("Sự kiện", name)
                    res_conf.metric("Độ tin cậy", f"{conf:.2%}")
                    
                    # Định dạng Log mới
                    new_log_card = f"""
                    <div class="detection-card">
                        <strong>[{curr_time}] {name}</strong><br>
                        Confidence: {conf:.2%}
                    </div>
                    """
                    
                    # Đẩy (Insert) thẻ log mới nhất LÊN ĐẦU DANH SÁCH (index 0)
                    st.session_state.event_logs.insert(0, new_log_card)
                    
                    # Giữ lại tối đa 30 logs cũ
                    if len(st.session_state.event_logs) > 30:
                        st.session_state.event_logs.pop()
                    
                    # Vẽ toàn bộ mảng Log
                    with log_placeholder.container():
                        st.markdown("".join(st.session_state.event_logs), unsafe_allow_html=True)
                
                # Nghỉ một chút để Streamlit kịp render
                time.sleep(0.01)

        except Exception as e:
            st.error(f"Lỗi luồng âm thanh: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            audio_p.terminate()
            st.session_state.running = False
            st.rerun()
else:
    st.warning("Vui lòng kiểm tra lại đường dẫn Model và Config ở Sidebar.")