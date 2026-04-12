"""
hWTF 함양량 계산 모듈 — 수정 버전
에러 원인: g = 1 - ths ** (1 / self.m) 에서 OverflowError
수정 내용:
  1. m ≤ 0 방지 (n > 1 강제)
  2. ths 범위 클램핑 (0~1 사이 보장)
  3. P_t > P_min 건조기 스킵
  4. 수치 안전 계산 wrapper
"""

import math
import pandas as pd
import numpy as np


class VanGenuchtenParams:
    """
    van Genuchten 비포화 수리 파라미터
    
    주의: n은 반드시 > 1.0 이어야 m = 1 - 1/n > 0 이 됩니다.
    n ≤ 1.0 이면 m ≤ 0 → (1/m) 지수 계산 불가 → OverflowError
    """
    def __init__(self, theta_r: float, theta_s: float,
                 alpha: float, n: float):
        self.theta_r = theta_r   # 잔류 함수비 (-)
        self.theta_s = theta_s   # 포화 함수비 (-)
        self.alpha   = alpha     # 역 공기 침입압 (1/cm 또는 1/m)
        
        # ── 핵심 수정 1: n > 1 보장 ──────────────────────────
        if n <= 1.0:
            raise ValueError(
                f"VG 파라미터 n={n} 은 반드시 1.0 초과여야 합니다. "
                f"n ≤ 1 이면 m = 1-1/n ≤ 0 이 되어 OverflowError 발생!"
            )
        self.n = n
        self.m = 1.0 - 1.0 / n   # m > 0 보장됨


class HWTFCalculator:
    """
    hWTF (Hydraulic Water Table Fluctuation) 함양량 산정 클래스
    박은규 교수 방법론 기반
    """

    def __init__(self, vg: VanGenuchtenParams, sy: float,
                 area_m2: float, p_min_mm: float = 1.0):
        """
        vg        : VanGenuchten 파라미터
        sy        : 충진공극률 (비산출률, 0~1)
        area_m2   : 대수층 면적 (m²)
        p_min_mm  : 유효 강수 최소값 (mm), 이 이하는 건조기로 처리
        """
        self.vg       = vg
        self.sy       = sy
        self.area     = area_m2
        self.p_min    = p_min_mm

    # ── 핵심 수정 2: ths 계산 안전 래퍼 ─────────────────────
    def _safe_ths(self, theta: float) -> float:
        """
        정규화된 함수비 Theta_s 계산
        반드시 (0, 1] 범위 안으로 클램핑
        """
        raw = (theta - self.vg.theta_r) / (self.vg.theta_s - self.vg.theta_r)
        # 0 이하 또는 1 초과 → 클램핑
        return max(1e-9, min(1.0, raw))

    # ── 핵심 수정 3: g 계산 안전 래퍼 ───────────────────────
    def _calc_g(self, ths: float) -> float:
        """
        g = 1 - ths^(1/m)
        ths가 0에 가까우면 ths^(1/m)도 0에 가까워지므로 g → 1
        ths=1 (포화) 이면 g = 0
        """
        try:
            # 1/m 지수: m > 0 은 __init__에서 보장됨
            exponent = 1.0 / self.vg.m
            # ths가 아주 작으면 큰 지수승도 0에 수렴하므로 안전
            val = ths ** exponent
            if not math.isfinite(val):
                return 0.0   # overflow 방어
            return 1.0 - val
        except (OverflowError, ZeroDivisionError, ValueError):
            return 0.0

    def calc_theta_ir(self, h_gw_m: float) -> float:
        """
        부정류 잔류함수비 θ_ir 계산
        h_gw_m : 지하수위 (m, 지표면 기준 깊이)
        """
        # 모세관압 (수두 형태, m)
        h_cap = h_gw_m  # 지표~지하수위 구간

        # VG 함수: theta(h) = theta_r + (theta_s - theta_r) / [1 + (alpha*h)^n]^m
        denom = (1.0 + (self.vg.alpha * h_cap) ** self.vg.n) ** self.vg.m
        theta = self.vg.theta_r + (self.vg.theta_s - self.vg.theta_r) / denom
        return theta

    def calc_recharge(self, p_mm: float, delta_h_m: float) -> float:
        """
        단일 시간 스텝 함양량 계산 (m³)
        
        p_mm      : 해당 시간 강수량 (mm)
        delta_h_m : 지하수위 상승량 (m), 양수 = 수위 상승
        
        ── 핵심 수정 4: P_min 이하 건조기 스킵 ──────────────
        """
        # 건조기: 강수 없으면 함양 없음
        if p_mm < self.p_min:
            return 0.0

        # 수위 상승이 없으면 함양 없음
        if delta_h_m <= 0:
            return 0.0

        # WTF (Water Table Fluctuation) 기반 함양량
        # R = Sy × ΔH × Area
        recharge_m3 = self.sy * delta_h_m * self.area
        return max(0.0, recharge_m3)

    def run_simulation(self, df: pd.DataFrame,
                       h_gw_col: str = "GW level (m)",
                       p_col: str = "Rainfall (mm)") -> pd.DataFrame:
        """
        전체 시계열 시뮬레이션

        df 필수 컬럼: Date, Rainfall (mm), GW level (m)
        반환: 함양량·함양률·누계 컬럼 추가된 DataFrame
        """
        df = df.copy()
        df["delta_H_m"]      = df[h_gw_col].diff().fillna(0)
        df["theta_ir"]       = df[h_gw_col].apply(self.calc_theta_ir)
        df["recharge_m3"]    = df.apply(
            lambda r: self.calc_recharge(r[p_col], r["delta_H_m"]), axis=1
        )
        df["recharge_mm"]    = df["recharge_m3"] / self.area * 1000
        df["cum_recharge_m3"] = df["recharge_m3"].cumsum()

        # 함양률 α = R(mm) / P(mm), P>0인 구간만
        mask = df[p_col] > self.p_min
        df["alpha_rate"] = 0.0
        df.loc[mask, "alpha_rate"] = (
            df.loc[mask, "recharge_mm"] / df.loc[mask, p_col]
        ).clip(0, 1)

        return df


# ─────────────────────────────────────────────────────────────
# 실행 예시 (Streamlit에서는 이 부분을 st.* 호출로 교체)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. CSV 로드
    df = pd.read_csv("/mnt/user-data/uploads/충적_hWTF_input.csv")
    print(f"데이터: {len(df)}행, 기간 {df['Date'].iloc[0]} ~ {df['Date'].iloc[-1]}")
    print(f"강수 이벤트: {(df['Rainfall (mm)'] > 0).sum()}개 시간")
    print(f"최대 강수: {df['Rainfall (mm)'].max()} mm")
    print()

    # 2. VG 파라미터 설정
    # ※ n=1.5 예시 — 실제 현장 토양 특성으로 교체 필요
    # n이 1.0 이하이면 ValueError 발생하여 미리 알 수 있음
    try:
        vg = VanGenuchtenParams(
            theta_r = 0.05,   # 잔류 함수비
            theta_s = 0.40,   # 포화 함수비
            alpha   = 0.01,   # 역 공기 침입압 (1/cm)
            n       = 1.5,    # 형상 지수 (반드시 > 1)
        )
        print(f"VG 파라미터: n={vg.n}, m={vg.m:.4f} ✓")
    except ValueError as e:
        print(f"파라미터 오류: {e}")
        exit(1)

    # 3. 계산기 초기화
    calc = HWTFCalculator(
        vg       = vg,
        sy       = 0.15,        # 충진공극률
        area_m2  = 50_000,      # 대수층 면적 (m²)
        p_min_mm = 1.0,         # 유효 강수 최솟값 (mm)
    )

    # 4. 시뮬레이션 실행
    result = calc.run_simulation(df)

    # 5. 결과 요약
    rain_events = result[result["Rainfall (mm)"] > 1.0]
    print("=== 시뮬레이션 결과 ===")
    print(f"총 함양량:    {result['recharge_m3'].sum():,.1f} m³")
    print(f"누계 함양량:  {result['cum_recharge_m3'].iloc[-1]:,.1f} m³")
    if len(rain_events) > 0:
        print(f"평균 함양률:  {rain_events['alpha_rate'].mean():.3f} (강수 이벤트 기간)")
    print()
    print("강수 이벤트 상세:")
    print(rain_events[["Date","Rainfall (mm)","delta_H_m",
                        "recharge_m3","alpha_rate"]].to_string(index=False))
