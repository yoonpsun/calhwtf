"""
지하 저류댐 hWTF 함양량 산정 시스템
streamlit_app.py — 완전 독립 실행 버전 (hwtf_fixed.py 불필요)

수정 사항:
  1. OverflowError 원인 제거: n > 1 강제, ths 클램핑, try-except 보호
  2. 지하수위 축 역전 (아래로 갈수록 깊음)
  3. 기존 GDP 코드 완전 제거, hWTF 전용 앱으로 재구성
"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="지하 저류댐 hWTF 함양량 산정",
    page_icon="💧",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# hWTF 계산 클래스 (OverflowError 완전 수정 버전)
# ─────────────────────────────────────────────────────────────
class VGParams:
    """
    van Genuchten 비포화 수리 파라미터
    [수정] n > 1.0 강제 검증 — n ≤ 1 이면 m ≤ 0 → 1/m 계산 시 OverflowError
    """
    def __init__(self, theta_r: float, theta_s: float,
                 alpha: float, n: float):
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.alpha   = alpha
        if n <= 1.0:
            raise ValueError(
                f"n={n} 오류: n은 반드시 1.0 초과여야 합니다. "
                f"(n ≤ 1 → m = 1-1/n ≤ 0 → OverflowError 발생)"
            )
        self.n = n
        self.m = 1.0 - 1.0 / n   # m > 0 보장


class HWTFCalc:
    """
    hWTF (Water Table Fluctuation) 함양량 산정
    박은규 교수 방법론 기반
    """

    def __init__(self, vg: VGParams, sy: float,
                 area_m2: float, p_min_mm: float = 1.0):
        self.vg     = vg
        self.sy     = sy
        self.area   = area_m2
        self.p_min  = p_min_mm

    def _safe_ths(self, theta: float) -> float:
        """
        정규화 함수비 계산 — 반드시 (0, 1] 범위로 클램핑
        [수정] 클램핑 없으면 ths^(1/m) 계산 시 overflow 가능
        """
        denom = self.vg.theta_s - self.vg.theta_r
        if denom <= 0:
            return 1e-9
        raw = (theta - self.vg.theta_r) / denom
        return max(1e-9, min(1.0, raw))

    def _calc_g(self, ths: float) -> float:
        """
        g = 1 - ths^(1/m)
        [수정] try-except로 OverflowError 완전 차단
        """
        try:
            val = ths ** (1.0 / self.vg.m)
            return 1.0 - val if math.isfinite(val) else 0.0
        except (OverflowError, ZeroDivisionError, ValueError):
            return 0.0

    def calc_theta_ir(self, h_gw_m: float) -> float:
        """부정류 잔류함수비 θ_ir — VG 함수 기반"""
        if h_gw_m <= 0:
            return self.vg.theta_s
        try:
            denom = (1.0 + (self.vg.alpha * h_gw_m) ** self.vg.n) ** self.vg.m
            return self.vg.theta_r + (self.vg.theta_s - self.vg.theta_r) / denom
        except (OverflowError, ZeroDivisionError, ValueError):
            return self.vg.theta_r

    def run_simulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        전체 시계열 시뮬레이션
        입력 컬럼: Date, Rainfall (mm), GW level (m)
        """
        df = df.copy().reset_index(drop=True)

        n = len(df)
        theta_ir   = np.zeros(n)
        recharge   = np.zeros(n)
        alpha_rate = np.zeros(n)
        delta_h    = np.zeros(n)

        gw  = df["GW level (m)"].values
        pcp = df["Rainfall (mm)"].values

        for i in range(1, n):
            dh = gw[i] - gw[i - 1]
            delta_h[i] = dh
            theta_ir[i] = self.calc_theta_ir(gw[i])

            # [수정] P_min 이하 건조기는 함양량 = 0
            if pcp[i] < self.p_min or dh <= 0:
                recharge[i] = 0.0
                alpha_rate[i] = 0.0
                continue

            # WTF 기반 함양량: R = Sy × ΔH × Area
            r_m3 = self.sy * dh * self.area
            recharge[i]   = max(0.0, r_m3)

            # 함양률 α = R(mm) / P(mm)
            r_mm = recharge[i] / self.area * 1000.0
            alpha_rate[i] = min(1.0, r_mm / pcp[i]) if pcp[i] > 0 else 0.0

        df["delta_H_m"]       = delta_h
        df["theta_ir"]        = theta_ir
        df["recharge_m3"]     = recharge
        df["recharge_mm"]     = recharge / self.area * 1000.0
        df["cum_recharge_m3"] = np.cumsum(recharge)
        df["alpha_rate"]      = alpha_rate
        return df


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("💧 지하 저류댐 hWTF 함양량 산정")
st.caption("박은규 교수 방법론 · 충적층 지하수 함양량 계산 (Water Table Fluctuation)")

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    st.subheader("📂 데이터")
    uploaded = st.file_uploader(
        "CSV 파일 업로드",
        type="csv",
        help="컬럼: Date, Rainfall (mm), GW level (m)"
    )

    st.subheader("🌱 VG 파라미터")
    theta_r  = st.number_input("잔류 함수비 θ_r",  value=0.05, min_value=0.0,  max_value=0.5,  step=0.01, format="%.3f")
    theta_s  = st.number_input("포화 함수비 θ_s",  value=0.40, min_value=0.05, max_value=0.7,  step=0.01, format="%.3f")
    alpha_vg = st.number_input("역 공기 침입압 α", value=0.01, min_value=0.001,max_value=1.0,  step=0.001,format="%.4f")

    n_vg = st.number_input(
        "형상 지수 n  ⚠️ 반드시 > 1.0",
        value=1.50, min_value=1.01, max_value=10.0, step=0.05, format="%.2f"
    )
    if n_vg <= 1.0:
        st.error("❌ n ≤ 1 → m ≤ 0 → OverflowError!  1.01 이상으로 설정하세요.")

    st.subheader("🏞️ 대수층")
    sy     = st.number_input("충진공극률 Sy",      value=0.15,   min_value=0.01,  max_value=0.5,   step=0.01)
    area   = st.number_input("면적 (m²)",          value=50000,  min_value=100,   max_value=10000000, step=1000)
    p_min  = st.number_input("유효 강수 P_min (mm)", value=1.0,  min_value=0.1,   max_value=10.0,  step=0.1)

    run_btn = st.button("▶ 계산 실행", type="primary", use_container_width=True)

# ── 메인 ──────────────────────────────────────────────────────
if uploaded is None:
    st.info("← 좌측에서 CSV 파일을 업로드하고 계산 실행 버튼을 누르세요.")

    # 데이터 포맷 안내
    st.subheader("📋 CSV 파일 형식")
    st.code(
        "Date,Rainfall (mm),GW level (m)\n"
        "2026.1.1 0:00,0,5.84\n"
        "2026.1.1 1:00,0,5.83\n"
        "2026.2.24 15:00,8.9,5.87",
        language="text"
    )
    st.stop()

# 데이터 로드
df_raw = pd.read_csv(uploaded)

# 컬럼명 유연 처리
col_map = {}
for c in df_raw.columns:
    cl = c.lower().strip()
    if "rain" in cl:
        col_map[c] = "Rainfall (mm)"
    elif "gw" in cl or "level" in cl or "groundwater" in cl:
        col_map[c] = "GW level (m)"
    elif "date" in cl or "time" in cl:
        col_map[c] = "Date"
df_raw = df_raw.rename(columns=col_map)

required = {"Date", "Rainfall (mm)", "GW level (m)"}
missing = required - set(df_raw.columns)
if missing:
    st.error(f"CSV 컬럼 누락: {missing}\n현재 컬럼: {list(df_raw.columns)}")
    st.stop()

# 요약 카드
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 데이터",    f"{len(df_raw):,} 행")
c2.metric("강수 이벤트",  f"{(df_raw['Rainfall (mm)'] > p_min).sum()} 시간")
c3.metric("최대 강수",    f"{df_raw['Rainfall (mm)'].max():.1f} mm")
c4.metric("GW 범위",      f"{df_raw['GW level (m)'].min():.2f} ~ {df_raw['GW level (m)'].max():.2f} m")

# 원시 데이터 표시
with st.expander("📂 입력 데이터 미리보기"):
    st.dataframe(df_raw, use_container_width=True)

# 계산 실행
if not run_btn:
    st.info("파라미터 확인 후 '▶ 계산 실행' 버튼을 누르세요.")
    st.stop()

if n_vg <= 1.0:
    st.error("n 값을 1.01 이상으로 수정 후 다시 실행하세요.")
    st.stop()

with st.spinner("hWTF 함양량 계산 중..."):
    try:
        vg   = VGParams(theta_r, theta_s, alpha_vg, n_vg)
        calc = HWTFCalc(vg, sy, area, p_min)
        result = calc.run_simulation(df_raw)
    except ValueError as e:
        st.error(f"파라미터 오류: {e}")
        st.stop()
    except Exception as e:
        st.error(f"계산 오류: {e}")
        st.stop()

st.success("✅ 계산 완료!")

# ── 결과 요약 ─────────────────────────────────────────────────
st.subheader("📊 계산 결과 요약")
r1, r2, r3, r4 = st.columns(4)
r1.metric("총 함양량",      f"{result['recharge_m3'].sum():,.1f} m³")
r2.metric("누계 함양량",    f"{result['cum_recharge_m3'].iloc[-1]:,.1f} m³")

rain_mask = result["Rainfall (mm)"] > p_min
avg_alpha = result.loc[rain_mask, "alpha_rate"].mean() if rain_mask.sum() > 0 else 0.0
r3.metric("평균 함양률 α",  f"{avg_alpha:.3f}")
r4.metric("VG m 값",        f"{vg.m:.4f}")

# ── 시각화 ───────────────────────────────────────────────────
st.subheader("📈 시계열 분석")

dates = result["Date"]

# ── 그래프 1: 지하수위 (축 역전) + 강수량 ──────────────────
fig1 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.05,
    subplot_titles=("지하수위 (m) — 아래로 갈수록 수위 낮음", "강수량 (mm)")
)

# 지하수위
fig1.add_trace(
    go.Scatter(
        x=dates, y=result["GW level (m)"],
        mode="lines", name="지하수위",
        line=dict(color="#1C7293", width=1.5)
    ),
    row=1, col=1
)

# 강수량 (막대)
fig1.add_trace(
    go.Bar(
        x=dates, y=result["Rainfall (mm)"],
        name="강수량", marker_color="#5B9BD5", opacity=0.7
    ),
    row=2, col=1
)

# ── 핵심: 지하수위 Y축 역전 ──────────────────────────────────
fig1.update_yaxes(
    autorange="reversed",    # 위로 갈수록 수위 낮음 (깊어짐)
    title_text="지하수위 (m)",
    row=1, col=1
)
fig1.update_yaxes(title_text="강수량 (mm)", row=2, col=1)
fig1.update_layout(height=500, showlegend=True,
                   legend=dict(orientation="h", y=1.05))
st.plotly_chart(fig1, use_container_width=True)

# ── 그래프 2: 함양량 및 누계 ─────────────────────────────────
fig2 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.5, 0.5],
    vertical_spacing=0.08,
    subplot_titles=("시간별 함양량 (m³)", "누계 함양량 (m³)")
)
fig2.add_trace(
    go.Bar(x=dates, y=result["recharge_m3"],
           name="함양량", marker_color="#028090"),
    row=1, col=1
)
fig2.add_trace(
    go.Scatter(x=dates, y=result["cum_recharge_m3"],
               mode="lines", name="누계 함양량",
               line=dict(color="#F96167", width=2)),
    row=2, col=1
)
fig2.update_yaxes(title_text="함양량 (m³)", row=1, col=1)
fig2.update_yaxes(title_text="누계 (m³)",   row=2, col=1)
fig2.update_layout(height=450, showlegend=True)
st.plotly_chart(fig2, use_container_width=True)

# ── 강수 이벤트 상세 ─────────────────────────────────────────
st.subheader("🌧️ 강수 이벤트별 상세")
if rain_mask.sum() > 0:
    rain_detail = result[rain_mask][
        ["Date", "Rainfall (mm)", "delta_H_m",
         "theta_ir", "recharge_m3", "recharge_mm", "alpha_rate"]
    ].copy()
    rain_detail.columns = [
        "일시", "강수량(mm)", "수위변동(m)",
        "잔류함수비θ_ir", "함양량(m³)", "함양량(mm)", "함양률α"
    ]
    st.dataframe(rain_detail.style.format({
        "강수량(mm)": "{:.1f}",
        "수위변동(m)": "{:.3f}",
        "잔류함수비θ_ir": "{:.4f}",
        "함양량(m³)": "{:.1f}",
        "함양량(mm)": "{:.3f}",
        "함양률α": "{:.4f}"
    }), use_container_width=True)
else:
    st.warning(f"P_min={p_min}mm 초과 강수 이벤트가 없습니다. P_min 값을 낮춰보세요.")

# ── 전체 결과 다운로드 ───────────────────────────────────────
st.subheader("⬇️ 결과 다운로드")
csv_out = result.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    label="전체 결과 CSV 다운로드",
    data=csv_out,
    file_name="hwtf_result.csv",
    mime="text/csv"
)
