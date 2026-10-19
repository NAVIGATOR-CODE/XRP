import streamlit as st
import pandas as pd
import numpy as np
import requests
from google import genai
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(
    page_title="My AI Agent Portal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 1. 구글 시트 연동 및 데이터 자동 기록 함수
# ---------------------------------------------------------
def save_data_to_google_sheet(df_latest):
    try:
        # Streamlit Secrets에 저장된 구글 서비스 계정 인증 정보 불러오기
        if "gcp_service_account" not in st.secrets:
            return "구글 시트 인증 정보(gcp_service_account)가 Streamlit Secrets에 설정되지 않았습니다."
        
        credentials_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
        gc = gspread.authorize(credentials)
        
        # 구글 시트 파일 열기 (이름: 'XRP_Quant_Database')
        sheet_name = "XRP_Quant_Database"
        try:
            spreadsheet = gc.open(sheet_name)
        except gspread.SpreadsheetNotFound:
            # 시트가 없으면 자동으로 생성
            spreadsheet = gc.create(sheet_name)
            
        worksheet = spreadsheet.get_worksheet(0)
        
        # 첫 행(헤더)이 비어있다면 컬럼 이름 작성
        if not worksheet.get_all_values():
            headers = ["date", "close", "volume", "RSI_14", "MFI_14", "OBV", "NVT_Proxy", "Signal"]
            worksheet.append_row(headers)
            
        # 오늘 날짜 데이터 추출 및 포맷팅
        row_data = [
            str(df_latest["date"]),
            float(df_latest["close"]),
            float(df_latest["volume"]),
            float(df_latest["RSI_14"]),
            float(df_latest["MFI_14"]),
            float(df_latest["OBV"]),
            float(df_latest["NVT_Proxy"]),
            str(df_latest["Signal"])
        ]
        
        # 중복 날짜가 아니라면 시트에 행 추가 (Append)
        existing_records = worksheet.get_all_records()
        existing_dates = [str(r.get("date")) for r in existing_records]
        
        if row_data[0] not in existing_dates:
            worksheet.append_row(row_data)
            return f"✅ 구글 시트('{sheet_name}')에 오늘자 데이터가 성공적으로 기록되었습니다!"
        else:
            return "📌 오늘자 데이터는 이미 구글 시트에 기록되어 있습니다."
            
    except Exception as e:
        return f"⚠️ 구글 시트 연동 중 오류 발생: {str(e)}"

# ---------------------------------------------------------
# 2. 수급 지표 계산 함수
# ---------------------------------------------------------
def calculate_technicals(df):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = (100 - (100 / (1 + rs))).round(2)

    tp = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["volume"]
    pos_flow = np.where(tp > tp.shift(1), rmf, 0)
    neg_flow = np.where(tp < tp.shift(1), rmf, 0)
    pos_mf = pd.Series(pos_flow, index=df.index).rolling(window=14).sum()
    neg_mf = pd.Series(neg_flow, index=df.index).rolling(window=14).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    df["MFI_14"] = (100 - (100 / (1 + mfr))).round(2)

    obv_change = np.where(df["close"] > df["close"].shift(1), df["volume"],
                 np.where(df["close"] < df["close"].shift(1), -df["volume"], 0))
    df["OBV"] = pd.Series(obv_change, index=df.index).cumsum()
    df["OBV_MA20"] = df["OBV"].rolling(window=20).mean()

    df["Signal"] = np.where(
        (df["OBV"] > df["OBV_MA20"]) & (df["MFI_14"] < 45), "강력 매집 (저점 유입)",
        np.where(df["OBV"] > df["OBV_MA20"], "매집 우위 (상승 추세)", "수급 분산 (관망/조정)")
    )
    return df

# ---------------------------------------------------------
# 3. 데이터 수집기
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_6months_xrp_data():
    kraken_url = "https://api.kraken.com/0/public/OHLC?pair=XRPUSD&interval=1440"
    res = requests.get(kraken_url, timeout=10).json()
    ohlc_list = list(res["result"].values())[0]
    
    df = pd.DataFrame(ohlc_list, columns=[
        "time", "open", "high", "low", "close", "vwap", "volume", "count"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["date"] = pd.to_datetime(df["time"], unit="s").dt.strftime("%Y-%m-%d")
    
    df = calculate_technicals(df).tail(180).reset_index(drop=True)

    xrpl_info = {"existing_supply": 99987000000, "burned_xrp": 13000000, "ledger_seq": 0}
    try:
        xrpl_res = requests.post(
            "https://s1.ripple.com:51234/",
            json={"method": "ledger", "params": [{"ledger_index": "validated"}]},
            timeout=10
        ).json()
        ledger_data = xrpl_res.get("result", {}).get("ledger", {})
        total_coins = int(ledger_data.get("total_coins", 99987000000000000)) / 1_000_000
        xrpl_info["existing_supply"] = round(total_coins, 0)
        xrpl_info["burned_xrp"] = round(100_000_000_000 - total_coins, 0)
        xrpl_info["ledger_seq"] = ledger_data.get("ledger_index", 0)
    except Exception:
        pass

    est_circulating = 56_800_000_000
    df["Est_MarketCap"] = df["close"] * est_circulating
    df["Daily_Dollar_Vol"] = df["close"] * df["volume"] * 15
    df["NVT_Proxy"] = (df["Est_MarketCap"] / df["Daily_Dollar_Vol"].replace(0, np.nan)).round(1)

    return df, xrpl_info

# ---------------------------------------------------------
# 사이드바 및 UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🤖 AI 에이전트 포털")
    agent_menu = st.radio(
        "실행할 에이전트를 선택하세요",
        [
            "📊 1. XRP 미래가치 & 온체인 에이전트",
            "📈 2. 주식 퀀트 (PER/PEG/수급) [확장예정]",
            "🏗️ 3. 스마트 업무/창작 에이전트 [확장예정]"
        ]
    )
    st.divider()
    default_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
    api_key = st.text_input("🔑 Gemini API Key", value=default_key, type="password")
    
    if st.button("🔄 6개월 데이터 강제 새로고침"):
        st.cache_data.clear()
        st.rerun()

if agent_menu.startswith("📊 1."):
    st.title("📊 XRP 6개월 수급 & 미래가치 밸류에이션 에이전트")
    
    with st.spinner("클라우드 서버가 최근 6개월 데이터 분석 및 구글 시트 연동 중입니다..."):
        df, xrpl_info = fetch_6months_xrp_data()
        latest = df.iloc[-1]
        prev_30d = df.iloc[-30]
        price_chg_30d = ((latest["close"] / prev_30d["close"]) - 1) * 100
        
        # 구글 드라이브(시트) 자동 기록 실행
        sheet_msg = save_data_to_google_sheet(latest)

    # 구글 시트 저장 결과 알림 표시
    st.toast(sheet_msg, icon="📊")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재 XRP 가격 (USD)", f"${latest['close']:.4f}", f"{price_chg_30d:+.1f}% (30일 전 대비)")
    c2.metric("MFI (14일 자금흐름지수)", f"{latest['MFI_14']}", "저점 매수세 유입" if latest['MFI_14'] < 40 else ("단기 과열" if latest['MFI_14'] > 70 else "중립 구간"))
    c3.metric("OBV 세력 수급 판독", latest["Signal"])
    c4.metric("XRPL 누적 소각량", f"{xrpl_info['burned_xrp']:,.0f} XRP", f"원장 #{xrpl_info['ledger_seq']}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["🧮 미래가치 시뮬레이터", "📈 6개월 수급 차트 & 구글 시트 연동", "🤖 AI 수석 애널리스트 리포트"])

    with tab1:
        st.subheader("🧮 XRPL 유틸리티 기반 적정가 계산기")
        col_a, col_b = st.columns([1, 1])
        
        with col_a:
            odl_vol = st.slider("① 연간 ODL 및 RLUSD 브릿지 결제 볼륨 (조 달러)", 0.5, 30.0, 5.0, step=0.5)
            velocity = st.slider("② XRP 연간 회전율 (Velocity)", 1, 50, 10)
            locked_xrp = st.slider("③ 기관 준비금 / AMM 풀 / ETF 락업 물량 (십억 개)", 5.0, 45.0, 20.0, step=1.0)
            rlusd_mcap = st.slider("④ RLUSD 스테이블코인 발행 시가총액 (십억 달러)", 1.0, 100.0, 10.0, step=1.0)

        with col_b:
            circulating_b = 56.8
            rlusd_locked_effect = (rlusd_mcap * 0.2) / max(0.5, latest["close"])
            effective_supply_b = max(2.0, circulating_b - locked_xrp - min(10.0, rlusd_locked_effect))
            
            required_mcap_b = (odl_vol * 1000) / velocity
            target_price = required_mcap_b / effective_supply_b
            upside_pct = ((target_price / latest["close"]) - 1) * 100

            st.info(f"📌 **유효 유통 공급량:** 약 **{effective_supply_b:.2f}B XRP**")
            st.info(f"📌 **요구 네트워크 가치(시총):** **${required_mcap_b:,.1f}B**")
            st.success(f"### 🎯 시뮬레이션 적정 가치: **${target_price:,.2f}**\n현재가 대비 기대 상승률: **{upside_pct:+,.0f}%**")

    with tab2:
        st.subheader("📈 최근 6개월 수급 차트 및 구글 시트 자동 연동 상태")
        st.success(sheet_msg)
        st.line_chart(df.set_index("date")[["close"]])
        st.line_chart(df.set_index("date")[["OBV", "OBV_MA20"]])
        
        with st.expander("📋 최근 6개월 일별 데이터 표(CSV) 보기 및 다운로드"):
            st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)
            csv_data = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 6개월 데이터 CSV 다운로드", data=csv_data, file_name="xrp_6m_quant.csv", mime="text/csv")

    with tab3:
        st.subheader("🤖 Gemini 퀀트 에이전트 심층 판독")
        user_question = st.text_input("추가 질문 (선택사항)", placeholder="예: 현재 MFI와 OBV 기준으로 단기 분할매수 타이밍인지 알려줘")
        
        if st.button("🚀 AI 수석 에이전트 리포트 생성", type="primary"):
            if not api_key:
                st.warning("왼쪽 사이드바에 Gemini API Key를 먼저 입력해 주세요.")
            else:
                with st.spinner("Gemini 에이전트가 분석 중입니다..."):
                    summary_payload = {
                        "기간": f"{df['date'].iloc[0]} ~ {latest['date']} (180일)",
                        "현재가": latest["close"],
                        "현재_RSI_14": latest["RSI_14"],
                        "현재_MFI_14": latest["MFI_14"],
                        "현재_수급시그널": latest["Signal"],
                        "시뮬레이션_목표가": round(target_price, 2)
                    }
                    prompt = f"""
                    당신은 냉철한 크립토 퀀트 애널리스트입니다. 데이터 요약: {summary_payload}
                    사용자 질문: {user_question if user_question else '전반적인 수급 및 미래가치 진단'}
                    1. 수급 진단 (OBV & MFI), 2. 미래가치 타당성, 3. 대응 전략으로 나누어 리포트를 작성하세요.
                    """
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(model="gemini-1.5-pro", contents=prompt)
                    st.markdown(response.text)
else:
    st.title(agent_menu)
    st.info("💡 향후 새로운 AI 에이전트 코드를 이어 붙일 수 있는 확장 슬롯입니다.")
