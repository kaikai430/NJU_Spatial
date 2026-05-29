from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.getisord import G_Local
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN, Queen
from libpysal.weights.util import attach_islands
from matplotlib.patches import Patch


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


DEFAULT_YEAR = 2022
DEFAULT_PERMUTATIONS = 999
LISA_LABELS = {
    0: "Not significant",
    1: "High-High",
    2: "Low-High",
    3: "Low-Low",
    4: "High-Low",
}
LISA_COLORS = {
    0: "#d9d9d9",
    1: "#d73027",
    2: "#91bfdb",
    3: "#4575b4",
    4: "#fdae61",
}
GISTAR_LABELS = {
    -3: "Cold spot 99%",
    -2: "Cold spot 95%",
    -1: "Cold spot 90%",
    0: "Not significant",
    1: "Hot spot 90%",
    2: "Hot spot 95%",
    3: "Hot spot 99%",
}
GISTAR_COLORS = {
    -3: "#2166ac",
    -2: "#67a9cf",
    -1: "#d1e5f0",
    0: "#f7f7f7",
    1: "#fddbc7",
    2: "#ef8a62",
    3: "#b2182b",
}
MUNICIPALITY_MAP = {
    "110000": "110100",
    "120000": "120100",
    "310000": "310100",
    "500000": "500100",
}


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def first_matching_column(columns: Iterable[str], keywords: list[str]) -> str:
    column_list = [str(col) for col in columns]
    lowered = {col: col.lower() for col in column_list}

    for keyword in keywords:
        for column in column_list:
            if keyword.lower() == lowered[column]:
                return column

    for keyword in keywords:
        for column in column_list:
            if keyword.lower() in lowered[column]:
                return column

    raise KeyError(f"未找到匹配列，候选关键词: {keywords}；现有列: {column_list}")


def choose_year(df: pd.DataFrame, year_col: str, year: int | None) -> int:
    numeric_year = pd.to_numeric(df[year_col], errors="coerce")
    available_years = sorted(int(item) for item in numeric_year.dropna().unique())
    if not available_years:
        raise ValueError("数据中没有可用年份。")
    return year if year is not None else available_years[-1]


def build_weight(gdf: gpd.GeoDataFrame, method: str) -> object:
    if method == "queen":
        weights = Queen.from_dataframe(gdf, use_index=True)
        if weights.islands:
            weights = attach_islands(weights, KNN.from_dataframe(gdf, k=1))
    elif method == "knn4":
        weights = KNN.from_dataframe(gdf, k=4)
    elif method == "knn8":
        weights = KNN.from_dataframe(gdf, k=8)
    else:
        raise ValueError(f"不支持的权重方法: {method}")

    weights.transform = "r"
    return weights


def build_weight_schemes(gdf: gpd.GeoDataFrame, primary_method: str) -> dict[str, object]:
    order = [primary_method] + [method for method in ["queen", "knn4", "knn8"] if method != primary_method]
    schemes: dict[str, object] = {}
    for method in order:
        schemes[method] = build_weight(gdf, method)
    return schemes


def is_standardized_hightech_csv(df: pd.DataFrame) -> bool:
    cols = {str(col) for col in df.columns}
    required = {"city", "行政区划代码", "year", "省份", "高新技术企业数"}
    return required.issubset(cols)


def standardize_hightech_csv(
    raw_csv_path: Path,
    reference_panel_path: Path,
    output_csv_path: Path,
    year: int = DEFAULT_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing = pd.read_csv(raw_csv_path, encoding="utf-8-sig")
    if is_standardized_hightech_csv(existing):
        panel_df = existing.copy()
    else:
        raw_df = pd.read_csv(raw_csv_path, header=None, encoding="utf-8-sig")
        reference_df = pd.read_csv(reference_panel_path, encoding="utf-8")

        ref_city_col = first_matching_column(reference_df.columns, ["city"])
        ref_code_col = first_matching_column(reference_df.columns, ["行政区划代码"])
        ref_province_col = first_matching_column(reference_df.columns, ["省份"])

        city_map = (
            reference_df[[ref_city_col, ref_code_col, ref_province_col]]
            .dropna(subset=[ref_city_col, ref_code_col])
            .drop_duplicates()
            .copy()
        )
        province_names = set(city_map[ref_province_col].astype(str))

        rows = raw_df.iloc[4:].copy()
        rows.columns = [
            "raw_city",
            "city_en",
            "高新技术企业数",
            "高新技术企业数_市辖区",
            "技术合同成交额_万元_全市",
            "技术合同成交额_万元_市辖区",
        ]
        rows["raw_city"] = rows["raw_city"].astype(str).str.strip()
        rows = rows[
            rows["raw_city"].notna()
            & rows["raw_city"].ne("nan")
            & ~rows["raw_city"].str.contains("续表", na=False)
            & rows["raw_city"].ne("城市")
        ].copy()

        rows["is_province"] = rows["raw_city"].isin(province_names)
        rows["省份"] = rows["raw_city"].where(rows["is_province"]).ffill()
        city_rows = rows.loc[~rows["is_province"]].copy()

        merged = city_rows.merge(
            city_map,
            left_on="raw_city",
            right_on=ref_city_col,
            how="left",
            suffixes=("_raw", ""),
            indicator=True,
        )
        unmatched_df = merged.loc[
            merged["_merge"] != "both",
            ["raw_city", "city_en", "省份_raw"],
        ].copy()
        unmatched_df = unmatched_df.rename(columns={"省份_raw": "省份"})
        matched_df = merged.loc[merged["_merge"] == "both"].copy()

        numeric_cols = [
            "高新技术企业数",
            "高新技术企业数_市辖区",
            "技术合同成交额_万元_全市",
            "技术合同成交额_万元_市辖区",
        ]
        for column in numeric_cols:
            matched_df[column] = pd.to_numeric(matched_df[column], errors="coerce")

        panel_df = pd.DataFrame(
            {
                "city": matched_df[ref_city_col].astype(str),
                "行政区划代码": matched_df[ref_code_col].astype(int).astype(str).str.zfill(6),
                "year": year,
                "省份": matched_df[ref_province_col].astype(str),
                "高新技术企业数": matched_df["高新技术企业数"],
                "高新技术企业数_市辖区": matched_df["高新技术企业数_市辖区"],
                "技术合同成交额_万元_全市": matched_df["技术合同成交额_万元_全市"],
                "技术合同成交额_万元_市辖区": matched_df["技术合同成交额_万元_市辖区"],
                "city_en": matched_df["city_en"].astype(str),
                "数据来源": "2-18 高新技术企业数和技术合同成交额",
            }
        ).sort_values(["省份", "行政区划代码"], ignore_index=True)

        unmatched_path = output_csv_path.with_name(f"{output_csv_path.stem}_未匹配城市.csv")
        unmatched_df.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

    panel_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    return panel_df, existing


def prepare_dataset(
    shp_path: Path,
    csv_path: Path,
    analysis_year: int | None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, str, str, str, int]:
    gdf = gpd.read_file(shp_path, encoding="utf-8")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    city_col = first_matching_column(df.columns, ["city", "城市", "地级市"])
    code_col = first_matching_column(df.columns, ["行政区划代码", "adcode", "地区代码"])
    year_col = first_matching_column(df.columns, ["year", "年份"])
    hightech_col = first_matching_column(
        df.columns,
        ["高新技术企业数", "高新技术企业数量", "高新技术企业", "高企数量", "高企数"],
    )

    target_year = choose_year(df, year_col, analysis_year)
    df = df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df[df[year_col] == target_year].copy()

    code_numeric = pd.to_numeric(df[code_col], errors="coerce")
    df["adcode"] = pd.Series(pd.NA, index=df.index, dtype="object")
    valid_code_mask = code_numeric.notna()
    df.loc[valid_code_mask, "adcode"] = (
        code_numeric.loc[valid_code_mask].astype(int).astype(str).str.zfill(6)
    )
    df = df.loc[df["adcode"].notna()].copy()
    df["adcode_mapped"] = df["adcode"].map(lambda code: MUNICIPALITY_MAP.get(code, code))
    df["hightech_value"] = pd.to_numeric(df[hightech_col], errors="coerce")

    gdf_merged = gdf.merge(
        df[[city_col, "adcode_mapped", "hightech_value"]],
        left_on="ct_adcode",
        right_on="adcode_mapped",
        how="left",
    )
    gdf_valid = gdf_merged.dropna(subset=["hightech_value"]).copy()
    gdf_valid = gdf_valid.reset_index(drop=True)

    if gdf_valid.empty:
        raise ValueError("合并后没有可用于空间自相关计算的数据，请检查代码列和企业数列。")

    return gdf_merged, gdf_valid, city_col, code_col, hightech_col, target_year


def classify_lisa(local_moran: Moran_Local, alpha: float) -> np.ndarray:
    labels = np.zeros(len(local_moran.q), dtype=int)
    significant = local_moran.p_sim < alpha
    labels[significant & (local_moran.q == 1)] = 1
    labels[significant & (local_moran.q == 2)] = 2
    labels[significant & (local_moran.q == 3)] = 3
    labels[significant & (local_moran.q == 4)] = 4
    return labels


def classify_gistar(z_scores: np.ndarray, p_values: np.ndarray) -> np.ndarray:
    classes = np.zeros(len(z_scores), dtype=int)
    classes[(z_scores > 0) & (p_values < 0.01)] = 3
    classes[(z_scores > 0) & (p_values >= 0.01) & (p_values < 0.05)] = 2
    classes[(z_scores > 0) & (p_values >= 0.05) & (p_values < 0.10)] = 1
    classes[(z_scores < 0) & (p_values < 0.01)] = -3
    classes[(z_scores < 0) & (p_values >= 0.01) & (p_values < 0.05)] = -2
    classes[(z_scores < 0) & (p_values >= 0.05) & (p_values < 0.10)] = -1
    return classes


def add_base_map(ax: plt.Axes, gdf_all: gpd.GeoDataFrame) -> None:
    gdf_all.plot(ax=ax, facecolor="#f8fafc", edgecolor="#d1d5db", linewidth=0.3)
    gdf_all.dissolve().boundary.plot(ax=ax, color="#4b5563", linewidth=0.8)
    ax.set_axis_off()


def plot_choropleth(
    gdf_all: gpd.GeoDataFrame,
    gdf_valid: gpd.GeoDataFrame,
    out_path: Path,
    year: int,
    value_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    add_base_map(ax, gdf_all)
    gdf_valid.plot(
        column="hightech_value",
        cmap="YlOrRd",
        linewidth=0.28,
        edgecolor="#5b616b",
        legend=True,
        legend_kwds={"label": value_label, "shrink": 0.72},
        ax=ax,
    )
    ax.set_title(f"{year}年中国地级市高新技术企业数空间分布", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_moran_simulation(moran: Moran, out_path: Path, year: int, weights_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(moran.sim, bins=30, color="#93c5fd", edgecolor="white")
    ax.axvline(moran.I, color="#dc2626", linewidth=2, label=f"Observed I = {moran.I:.4f}")
    ax.set_title(f"{year}年高新技术企业数 Moran's I 置换检验（{weights_name}）", fontsize=14)
    ax.set_xlabel("模拟 Moran's I")
    ax.set_ylabel("频数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_moran_scatter(
    gdf: gpd.GeoDataFrame,
    weights: object,
    out_path: Path,
    year: int,
    weights_name: str,
) -> None:
    y = gdf["hightech_value"].to_numpy(dtype=float)
    z = (y - y.mean()) / y.std(ddof=0)
    lag_z = np.asarray(weights.sparse @ z).ravel()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(z, lag_z, s=28, alpha=0.82, color="#2563eb", edgecolor="white", linewidth=0.4)
    slope, intercept = np.polyfit(z, lag_z, 1)
    x_limit = np.max(np.abs(z)) * 1.08
    y_limit = np.max(np.abs(lag_z)) * 1.08
    xs = np.linspace(-x_limit, x_limit, 200)
    ax.plot(xs, slope * xs + intercept, color="#dc2626", linewidth=2)
    ax.axhline(0, color="#666666", linewidth=1, linestyle="--")
    ax.axvline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{year}年高新技术企业数 Moran 散点图（{weights_name}）", fontsize=14)
    ax.set_xlabel("标准化高新技术企业数")
    ax.set_ylabel("空间滞后值")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lisa_cluster(
    gdf_all: gpd.GeoDataFrame,
    gdf_valid: gpd.GeoDataFrame,
    out_path: Path,
    year: int,
    weights_name: str,
    alpha: float,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    add_base_map(ax, gdf_all)
    for class_value in [1, 3, 2, 4, 0]:
        subset = gdf_valid[gdf_valid["lisa_class"] == class_value]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            facecolor=LISA_COLORS[class_value],
            edgecolor="#5b616b",
            linewidth=0.28,
            alpha=0.9,
        )

    handles = [
        Patch(facecolor=LISA_COLORS[key], edgecolor="#5b616b", label=LISA_LABELS[key])
        for key in [1, 3, 2, 4, 0]
        if (gdf_valid["lisa_class"] == key).any()
    ]
    ax.legend(handles=handles, loc="lower left", title=f"LISA（p < {alpha:.2f}）", frameon=True)
    ax.set_title(f"{year}年高新技术企业数 LISA 聚类图（{weights_name}）", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gistar_hotspots(
    gdf_all: gpd.GeoDataFrame,
    gdf_valid: gpd.GeoDataFrame,
    out_path: Path,
    year: int,
    weights_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    add_base_map(ax, gdf_all)
    for class_value in [3, 2, 1, -1, -2, -3, 0]:
        subset = gdf_valid[gdf_valid["gi_star_class"] == class_value]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            facecolor=GISTAR_COLORS[class_value],
            edgecolor="#5b616b",
            linewidth=0.28,
            alpha=0.9,
        )

    handles = [
        Patch(facecolor=GISTAR_COLORS[key], edgecolor="#5b616b", label=GISTAR_LABELS[key])
        for key in [3, 2, 1, -1, -2, -3, 0]
        if (gdf_valid["gi_star_class"] == key).any()
    ]
    ax.legend(handles=handles, loc="lower left", title="Getis-Ord Gi*", frameon=True)
    ax.set_title(f"{year}年高新技术企业数 Getis-Ord Gi* 热点图（{weights_name}）", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(global_results: pd.DataFrame, out_path: Path, year: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    bar_colors = ["#dc2626" if value > 0 else "#2563eb" for value in global_results["moran_i"]]
    bars = axes[0].bar(global_results["weights_method"], global_results["moran_i"], color=bar_colors)
    axes[0].axhline(0, color="#111827", linewidth=0.8)
    axes[0].set_ylabel("Moran's I")
    axes[0].set_title(f"{year}年不同权重下的 Moran's I")
    for bar, value in zip(bars, global_results["moran_i"], strict=False):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    p_values = global_results["p_value"].clip(lower=1e-4)
    bars = axes[1].bar(global_results["weights_method"], p_values, color="#60a5fa")
    axes[1].axhline(0.05, color="#dc2626", linewidth=1.6, linestyle="--", label="0.05")
    axes[1].axhline(0.01, color="#f59e0b", linewidth=1.2, linestyle="--", label="0.01")
    axes[1].set_ylabel("Permutation p-value")
    axes[1].set_title(f"{year}年不同权重下的显著性对比")
    axes[1].legend()
    for bar, value, raw in zip(bars, p_values, global_results["p_value"], strict=False):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.002,
            f"{raw:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_local_summary(
    gdf: gpd.GeoDataFrame,
    year: int,
    primary_weights: str,
    lisa_alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_value, class_name in LISA_LABELS.items():
        rows.append(
            {
                "year": year,
                "analysis": "lisa",
                "weights_method": primary_weights,
                "alpha": lisa_alpha,
                "category": class_name,
                "count": int((gdf["lisa_class"] == class_value).sum()),
            }
        )

    for class_value, class_name in GISTAR_LABELS.items():
        rows.append(
            {
                "year": year,
                "analysis": "gi_star",
                "weights_method": primary_weights,
                "alpha": np.nan,
                "category": class_name,
                "count": int((gdf["gi_star_class"] == class_value).sum()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="整理高新技术企业 CSV，并完成全局/局部空间自相关、热点分析与稳健性检验。"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/china/高新技术企业数_标准化.csv"),
        help="高新技术企业 CSV 文件路径。",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("data/china/地级市数据.csv"),
        help="用于补充行政区划代码和省份的参考 CSV。",
    )
    parser.add_argument(
        "--shp",
        type=Path,
        default=Path("data/china/city.shp"),
        help="地级市矢量边界文件路径。",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="分析年份；若原始高新技术企业表没有年份，将写入该年份。",
    )
    parser.add_argument(
        "--weights",
        choices=["queen", "knn4", "knn8"],
        default="queen",
        help="主分析的空间权重构建方式，稳健性检验会同步比较其他权重。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hightech_moran"),
        help="图表和结果文件输出目录。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="置换检验随机种子，默认 42。",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help="置换检验次数，默认 999。",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="LISA 显著性水平，默认 0.05。",
    )
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    standardized_df, original_df = standardize_hightech_csv(
        raw_csv_path=args.csv,
        reference_panel_path=args.reference_csv,
        output_csv_path=args.csv,
        year=args.year,
    )

    gdf_all, gdf, city_col, code_col, hightech_col, target_year = prepare_dataset(
        args.shp,
        args.csv,
        args.year,
    )

    values = gdf["hightech_value"].to_numpy(dtype=float)
    weight_schemes = build_weight_schemes(gdf, args.weights)

    global_rows: list[dict[str, object]] = []
    moran_results: dict[str, Moran] = {}
    for method_name, weights in weight_schemes.items():
        moran = Moran(values, weights, permutations=args.permutations)
        moran_results[method_name] = moran
        global_rows.append(
            {
                "year": target_year,
                "weights_method": method_name,
                "is_primary": method_name == args.weights,
                "n_cities": len(gdf),
                "moran_i": moran.I,
                "expected_i": moran.EI,
                "z_score": moran.z_sim,
                "p_value": moran.p_sim,
                "seed": args.seed,
                "permutations": args.permutations,
                "hightech_column": hightech_col,
                "city_column": city_col,
                "code_column": code_col,
                "standardized_rows": len(standardized_df),
                "raw_rows": len(original_df),
                "n_islands": len(getattr(weights, "islands", [])),
            }
        )

    global_summary = pd.DataFrame(global_rows)
    global_summary_path = args.output_dir / "moran_summary.csv"
    global_summary.to_csv(global_summary_path, index=False, encoding="utf-8-sig")

    primary_weights = weight_schemes[args.weights]
    primary_moran = moran_results[args.weights]
    local_moran = Moran_Local(values, primary_weights, permutations=args.permutations)
    gi_star = G_Local(values, primary_weights, star=1.0, permutations=args.permutations)

    gdf["local_moran_i"] = local_moran.Is
    gdf["local_moran_p"] = local_moran.p_sim
    gdf["lisa_quadrant"] = local_moran.q
    gdf["lisa_class"] = classify_lisa(local_moran, args.alpha)
    gdf["lisa_label"] = gdf["lisa_class"].map(LISA_LABELS)
    gdf["gi_star_z"] = gi_star.Zs
    gdf["gi_star_p"] = gi_star.p_sim
    gdf["gi_star_class"] = classify_gistar(gdf["gi_star_z"].to_numpy(), gdf["gi_star_p"].to_numpy())
    gdf["gi_star_label"] = gdf["gi_star_class"].map(GISTAR_LABELS)

    for method_name, weights in weight_schemes.items():
        gdf[f"spatial_lag_{method_name}"] = np.asarray(weights.sparse @ values).ravel()

    local_summary = create_local_summary(gdf, target_year, args.weights, args.alpha)
    local_summary_path = args.output_dir / "local_spatial_summary.csv"
    local_summary.to_csv(local_summary_path, index=False, encoding="utf-8-sig")

    local_results_path = args.output_dir / "local_spatial_results.csv"
    gdf.drop(columns="geometry").to_csv(local_results_path, index=False, encoding="utf-8-sig")

    merged_path = args.output_dir / "hightech_moran_merged.geojson"
    gdf.to_file(merged_path, driver="GeoJSON", encoding="utf-8")

    plot_choropleth(
        gdf_all,
        gdf,
        args.output_dir / "hightech_spatial_distribution.png",
        target_year,
        hightech_col,
    )
    plot_moran_simulation(
        primary_moran,
        args.output_dir / "moran_simulation_histogram.png",
        target_year,
        args.weights,
    )
    plot_moran_scatter(
        gdf,
        primary_weights,
        args.output_dir / "moran_scatter.png",
        target_year,
        args.weights,
    )
    plot_lisa_cluster(
        gdf_all,
        gdf,
        args.output_dir / "lisa_cluster_map.png",
        target_year,
        args.weights,
        args.alpha,
    )
    plot_gistar_hotspots(
        gdf_all,
        gdf,
        args.output_dir / "gi_star_hotspot_map.png",
        target_year,
        args.weights,
    )
    plot_robustness(
        global_summary[["weights_method", "moran_i", "p_value"]],
        args.output_dir / "robustness_comparison.png",
        target_year,
    )

    print("=" * 60)
    print("高新技术企业数空间自相关分析完成")
    print("=" * 60)
    print(f"分析年份: {target_year}")
    print(f"标准化后城市数: {len(standardized_df)}")
    print(f"参与空间分析的城市数: {len(gdf)}")
    print(f"企业数字段: {hightech_col}")
    print(f"主权重方法: {args.weights}")
    print(f"Moran's I: {primary_moran.I:.6f}")
    print(f"Expected I: {primary_moran.EI:.6f}")
    print(f"Z-score: {primary_moran.z_sim:.6f}")
    print(f"P-value: {primary_moran.p_sim:.6f}")
    print(f"标准化 CSV: {args.csv}")
    print(f"全局结果汇总: {global_summary_path}")
    print(f"局部结果汇总: {local_summary_path}")
    print(f"局部结果明细: {local_results_path}")
    print(f"空间分布图: {args.output_dir / 'hightech_spatial_distribution.png'}")
    print(f"置换检验图: {args.output_dir / 'moran_simulation_histogram.png'}")
    print(f"Moran 散点图: {args.output_dir / 'moran_scatter.png'}")
    print(f"LISA 聚类图: {args.output_dir / 'lisa_cluster_map.png'}")
    print(f"Gi* 热点图: {args.output_dir / 'gi_star_hotspot_map.png'}")
    print(f"稳健性对比图: {args.output_dir / 'robustness_comparison.png'}")
    print(f"合并后空间数据: {merged_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
