# -*- coding: utf-8 -*-
"""지역별 외국인 관광객·숙박·의료관광 데이터 분석 및 군집화"""

import glob
import os
import folium
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 0. 환경 설정 (로컬/Colab 공용 경로 기준)
# ---------------------------------------------------------
BASE_DIR = os.environ.get("DATA_DIR", "./data")
VISITOR_DIR = os.path.join(BASE_DIR, "외국인 방문자수 - 외부데이터")
LODGING_DIR = os.path.join(BASE_DIR, "숙박업")
MEDICAL_DIR = os.path.join(BASE_DIR, "의료관광 현황")


def setup_korean_font(font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    """환경에 한글 폰트가 있으면 적용합니다."""
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path, size=10).get_name()
        plt.rc("font", family=font_name)
    plt.rcParams["axes.unicode_minus"] = False


def add_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """기준년월일 컬럼으로부터 연/월/연월 파생 컬럼을 생성합니다."""
    df = df.copy()
    ym_str = df["기준년월일"].astype(str)
    df["연도"] = ym_str.str[:4]
    df["월"] = ym_str.str[4:] + "월"
    df["연월"] = df["연도"] + "-" + df["월"]
    return df


def plot_monthly_trend(df: pd.DataFrame, title: str):
    """월별 방문자 수 추이 선그래프를 출력합니다."""
    plt.figure(figsize=(12, 6))
    plt.plot(df["연월"], df["방문자 수"], marker="o", linestyle="-", color="blue")
    plt.title(title, fontsize=16)
    plt.xlabel("연월", fontsize=12)
    plt.ylabel("방문자 수", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


setup_korean_font()

# ---------------------------------------------------------
# 1. 전국 및 시도별 월별 외국인 방문자 수 추이
# ---------------------------------------------------------
regions = {
    "all": ("all.csv", "월별 전국 외국인 방문자 수 추이"),
    "seoul": ("seoul.csv", "월별 서울 외국인 방문자 수 추이"),
    "gyeonggi": ("gyeonggi.csv", "월별 경기도 외국인 방문자 수 추이"),
    "gangwon": ("gangwon.csv", "월별 강원도 외국인 방문자 수 추이"),
    "incheon": ("incheon.csv", "월별 인천광역시 외국인 방문자 수 추이"),
    "chungnam": ("chungnam.csv", "월별 충청남도 외국인 방문자 수 추이"),
    "chungbuk": ("chungbuk.csv", "월별 충청북도 외국인 방문자 수 추이"),
    "sejong": ("sejong.csv", "월별 세종특별자치시 외국인 방문자 수 추이"),
    "daejeon": ("daejeon.csv", "월별 대전광역시 외국인 방문자 수 추이"),
    "jeonbuk": ("jeonbuk.csv", "월별 전라북도 외국인 방문자 수 추이"),
    "jeonnam": ("jeonnam.csv", "월별 전라남도 외국인 방문자 수 추이"),
    "daegu": ("daegu.csv", "월별 대구광역시 외국인 방문자 수 추이"),
    "gyeongbuk": ("gyeongbuk.csv", "월별 경상북도 외국인 방문자 수 추이"),
    "gyeongnam": ("gyeongnam.csv", "월별 경상남도 외국인 방문자 수 추이"),
    "ulsan": ("ulsan.csv", "월별 울산광역시 외국인 방문자 수 추이"),
    "busan": ("busan.csv", "월별 부산광역시 외국인 방문자 수 추이"),  # 원본 오타(ulsan.csv) 수정
    "jeju": ("jeju.csv", "월별 제주특별자치도 외국인 방문자 수 추이"),
}

region_dfs = {}
for key, (filename, chart_title) in regions.items():
    file_path = os.path.join(VISITOR_DIR, filename)
    df_region = pd.read_csv(file_path, encoding="utf-8")
    df_region = add_year_month(df_region)  # jeju 연월 참조 오타 수정
    region_dfs[key] = df_region
    plot_monthly_trend(df_region, chart_title)

# 지역별 방문자 누적 수 순위
visitor_files = [
    os.path.join(VISITOR_DIR, fname)
    for key, (fname, _) in regions.items()
    if key != "all"
]
df_visitors_all = pd.concat([pd.read_csv(f, encoding="utf-8") for f in visitor_files], ignore_index=True)
df_visitor_sorted = (
    df_visitors_all.groupby("지역")["방문자 수"]
    .sum()
    .reset_index()
    .sort_values(by="방문자 수", ascending=False)
    .reset_index(drop=True)
)
df_visitor_sorted.index = df_visitor_sorted.index + 1
display(df_visitor_sorted)

# ---------------------------------------------------------
# 2. 지역별 숙박업소 현황 및 지도 시각화
# ---------------------------------------------------------
lodging_csv_files = glob.glob(os.path.join(LODGING_DIR, "*.csv"))
lodging_list = []
for file in lodging_csv_files:
    region_name = os.path.basename(file).replace(".csv", "")
    temp_df = pd.read_csv(file)
    # 지역별 숙박영업현황수 컬럼이 있는 지역 파일만 병합
    if "숙박영업현황수" in temp_df.columns:
        temp_df["지역"] = region_name
        lodging_list.append(temp_df)

df_lodging_all = pd.concat(lodging_list, ignore_index=True)
df_lodging_sum = (
    df_lodging_all.groupby("지역")["숙박영업현황수"]
    .sum()
    .reset_index()
    .rename(columns={"숙박영업현황수": "총숙박업소수"})
    .sort_values(by="총숙박업소수", ascending=False)
    .reset_index(drop=True)
)

region_coords = {
    "seoul": [37.5665, 126.9780],
    "busan": [35.1796, 129.0756],
    "daegu": [35.8714, 128.6014],
    "incheon": [37.4563, 126.7052],
    "gwangju": [35.1595, 126.8526],
    "daejeon": [36.3504, 127.3845],
    "ulsan": [35.5384, 129.3114],
    "sejong": [36.4800, 127.2890],
    "gyeonggi": [37.2752, 127.0095],
    "gangwon": [37.8228, 128.1555],
    "chungbuk": [36.6357, 127.4913],
    "chungnam": [36.5184, 126.8000],
    "jeonbuk": [35.8205, 127.1087],
    "jeonnam": [34.8160, 126.4630],
    "gyeongbuk": [36.4919, 128.8889],
    "gyeongnam": [35.4606, 128.2132],
    "jeju": [33.4996, 126.5312],
}

df_lodging_sum["위도"] = df_lodging_sum["지역"].map(lambda x: region_coords.get(x, [None, None])[0])
df_lodging_sum["경도"] = df_lodging_sum["지역"].map(lambda x: region_coords.get(x, [None, None])[1])
df_map = df_lodging_sum.dropna(subset=["위도", "경도"]).reset_index(drop=True)

map_korea = folium.Map(location=[36.5, 127.8], zoom_start=7)
for _, row in df_map.iterrows():
    radius = max(row["총숙박업소수"] / 200, 10)
    folium.CircleMarker(
        location=[row["위도"], row["경도"]],
        radius=radius,
        popup=f"{row['지역']}<br>숙박업소: {row['총숙박업소수']}",
        color="gray",
        fill=True,
        fill_color="gray",
        fill_opacity=0.6,
    ).add_to(map_korea)

display(map_korea)

# ---------------------------------------------------------
# 3. 의료관광 추이 분석 및 K-Means 클러스터링
# ---------------------------------------------------------
hospital_country = pd.read_csv(os.path.join(MEDICAL_DIR, "country.csv"))
hospital_country_mean = hospital_country.groupby("기준연도")["외국인관광객수"].mean().reset_index()
hospital_country_sum = hospital_country.groupby("고객국가명")["외국인관광객수"].sum().reset_index()
hospital_country_total = hospital_country.groupby("기준연도")["외국인관광객수"].sum().reset_index()

plt.figure(figsize=(10, 12))
plt.subplot(3, 1, 1)
sns.barplot(data=hospital_country_mean, x="기준연도", y="외국인관광객수", palette="Blues_r")
plt.title("년도별 평균 의료 관광 외국인 수")
plt.xlabel("년도")
plt.ylabel("평균 관광객수")

plt.subplot(3, 1, 2)
sns.barplot(data=hospital_country_sum, x="고객국가명", y="외국인관광객수", palette="viridis")
plt.title("국가별 누적 의료 관광객수")
plt.xlabel("국가")
plt.ylabel("누적 관광객수")

plt.subplot(3, 1, 3)
sns.lineplot(data=hospital_country_total, x="기준연도", y="외국인관광객수", marker="o", linewidth=2)
plt.title("년도별 전체 의료 관광객수 변화")
plt.xlabel("년도")
plt.ylabel("전체 관광객수")
plt.tight_layout()
plt.show()

hospital_region = pd.read_csv(os.path.join(MEDICAL_DIR, "region.csv"))
region_mean = hospital_region.groupby("지역명")["외국인관광객수"].mean().reset_index()
region_trend = hospital_region.pivot(index="지역명", columns="기준연도", values="외국인관광객수")
region_total = hospital_region.groupby("지역명")["외국인관광객수"].sum().reset_index()

plt.figure(figsize=(10, 12))
plt.subplot(3, 1, 1)
sns.barplot(data=region_mean, x="지역명", y="외국인관광객수", palette="Blues_r")
plt.title("지역별 평균 의료 관광 외국인수")
plt.xlabel("지역")
plt.ylabel("평균 관광객수")

plt.subplot(3, 1, 2)
for col in region_trend.columns:
    plt.plot(region_trend.index, region_trend[col], marker="o", label=col)
plt.title("지역별 의료 관광 외국인수 변화(년도별)")
plt.xlabel("년도")
plt.ylabel("관광객수")
plt.legend(title="지역명")
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(3, 1, 3)
sns.barplot(data=region_total, x="지역명", y="외국인관광객수", palette="viridis")
plt.title("지역별 누적 의료 관광 외국인수")
plt.xlabel("지역")
plt.ylabel("누적 관광객수")
plt.tight_layout()
plt.show()

# 전년대비 증감률 및 연도별 전체 대비 비중
hospital_region = hospital_region.sort_values(["지역명", "기준연도"]).copy()
hospital_region["전년대비증감률"] = hospital_region.groupby("지역명")["외국인관광객수"].pct_change() * 100
total_by_year = hospital_region.groupby("기준연도")["외국인관광객수"].transform("sum")
hospital_region["전체대비비율"] = hospital_region["외국인관광객수"] / total_by_year

# K-Means 군집화
scaler = StandardScaler()
scaled_df = scaler.fit_transform(region_trend.fillna(0))

inertias = []
k_range = range(1, 6)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_df)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, "o-")
plt.xlabel("Number of clusters K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
region_trend["cluster"] = kmeans.fit_predict(scaled_df)
display(region_trend)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=region_trend[2022],
    y=region_trend[2024],
    hue=region_trend["cluster"],
    palette="Set2",
    s=200,
)
plt.title("지역별 의료 관광객 패턴 클러스터링")
plt.xlabel("2022 관광객수")
plt.ylabel("2024 관광객수")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()