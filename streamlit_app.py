"""
지하 저류댐 hWTF 함양량 산정 — Streamlit 앱 (수정 버전)
"""

import streamlit as st
import pandas as pd
import math
from pathlib import Path
from hwtf_fixed import VanGenuchtenParams, HWTFCalculator

st.set_page_config(
    page_title="지하 저류댐 hWTF 함양량 산정",
    page_icon="💧",
    layout="wide"
)

st.title("💧 지하 저류댐 hWTF 함양량 산정")
st.caption("박은규 교수 방법론 기반 · 충적층 지하수 함양량 계산")

# ── 사이드바: 파라미터 입력 ───────────────────────────────────
with st.sidebar:
    st.header("파라미터 설정")

    st.subheader("📂 데이터 업로드")
    uploaded = st.file_uploader("CSV 파일 (Date, Rainfall(mm), GW level(m))", type="csv")

    st.subheader("🌱 VG 파라미터")
    theta_r = st.number_input("잔류 함수비 θ_r", value=0.05, min_value=0.0, max_value=0.5, step=0.01)
    theta_s = st.number_input("포화 함수비 θ_s", value=0.40, min_value=0.1, max_value=0.6, step=0.01)
    alpha_vg = st.number_input("역 공기 침입압 α (1/cm)", value=0.01, min_value=0.001, step=0.001, format="%.4f")

    # ── 핵심: n > 1 안내 ──
    n_vg = st.number_input(
        "형상 지수 n (반드시 > 1.0)",
        value=1.5, min_value=1.01, max_value=5.0, step=0.05
    )
    if n_vg <= 1.0:
        st.error("⚠️ n은 1.0 초과여야 합니다! n ≤ 1 → m ≤ 0 → OverflowError 발생")

    st.subheader("🏞️ 대수층 설정")
    sy      = st.number_input("충진공극률 Sy", value=0.15, min_value=0.01, max_value=0.5, step=0.01)
    area    = st.number_input("대수층 면적 (m²)", value=50000, min_value=100, step=1000)
    p_min   = st.number_input("유효 강수 최솟값 P_min (mm)", value=1.0, min_value=0.1, step=0.1)

# ── 메인: 계산 및 결과 ────────────────────────────────────────
if uploaded is None:
    st.info("← 좌측에서 CSV 파일을 업로드하고 파라미터를 설정하세요.")
    st.stop()

# 데이터 로드
df = pd.read_csv(uploaded)
st.subheader("📋 입력 데이터")
col1, col2, col3 = st.columns(3)
col1.metric("총 데이터", f"{len(df):,} 행")
col2.metric("강수 이벤트", f"{(df['Rainfall (mm)'] > p_min).sum()} 시간")
col3.metric("최대 강수", f"{df['Rainfall (mm)'].max()} mm")
st.dataframe(df.tail(20), use_container_width=True)

# 파라미터 검증 후 계산
if n_vg <= 1.0:
    st.error("n 값을 1.0 초과로 수정 후 다시 실행하세요.")
    st.stop()

try:
    vg   = VanGenuchtenParams(theta_r, theta_s, alpha_vg, n_vg)
    calc = HWTFCalculator(vg, sy, area, p_min)
    result = calc.run_simulation(df)

except ValueError as e:
    st.error(f"파라미터 오류: {e}")
    st.stop()

except Exception as e:
    st.error(f"계산 오류: {e}")
    st.stop()

# ── 결과 출력 ─────────────────────────────────────────────────
st.subheader("📊 계산 결과")

c1, c2, c3, c4 = st.columns(4)
c1.metric("총 함양량",    f"{result['recharge_m3'].sum():,.1f} m³")
c2.metric("누계 함양량",  f"{result['cum_recharge_m3'].iloc[-1]:,.1f} m³")

rain_mask = result["Rainfall (mm)"] > p_min
if rain_mask.sum() > 0:
    avg_alpha = result.loc[rain_mask, "alpha_rate"].mean()
    c3.metric("평균 함양률", f"{avg_alpha:.3f}")
else:
    c3.metric("평균 함양률", "강수 없음")

c4.metric("VG m값", f"{vg.m:.4f}")

st.subheader("📈 지하수위 및 함양량 시계열")
st.line_chart(result.set_index("Date")[["GW level (m)"]])
st.line_chart(result.set_index("Date")[["recharge_m3", "cum_recharge_m3"]])

st.subheader("🌧️ 강수 이벤트별 상세")
rain_detail = result[result["Rainfall (mm)"] > p_min][
    ["Date", "Rainfall (mm)", "delta_H_m", "recharge_m3", "alpha_rate"]
]
st.dataframe(rain_detail, use_container_width=True)
