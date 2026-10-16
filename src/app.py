# app.py
import streamlit as st
from src import notion_rag_backend

st.set_page_config(page_title="Notion × LLM", page_icon="🧠", layout="wide")

st.title("🧠 Notion × LLM 助理")
st.markdown("問我任何與你的 Notion 筆記相關的問題吧！")

# 側邊欄設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # 檢索參數
    st.subheader("檢索設定")
    top_k = st.slider("檢索數量", 1, 10, 3, help="從知識庫中檢索幾筆相關內容")
    
    # LLM 參數
    st.subheader("LLM 設定")
    model_name = st.selectbox(
        "模型選擇",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
    temperature = st.slider("創造力", 0.0, 1.0, 0.3, 0.1, help="越高越有創意，越低越精確")
    
    # 顯示設定
    st.subheader("顯示設定")
    st.session_state.show_sources = st.checkbox("顯示資料來源", value=True)
    

# 初始化 session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 使用者輸入
user_query = st.chat_input("輸入你的問題...")

if user_query:
    with st.spinner("思考中..."):
        answer = notion_rag_backend.generate_answer(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# 顯示聊天紀錄
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])