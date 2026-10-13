"""Chuẩn hoá trên thư viện tham chiếu và truy vấn top-k tương đồng."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

GROUPS = {
    "pitch": [c for c in ("f0_hz_mean", "f0_hz_std", "f0_hz_min", "f0_hz_max", "f0_hz_range", "f0_hz_p25", "f0_hz_p50", "f0_hz_p75")],
    "mfcc": [f"mfcc_{i:02d}_{s}" for i in range(1, 14) for s in ("mean", "std")],
    "formant": [f"formant_f{k}_hz_{s}" for k in (1,2,3) for s in ("mean","std")],
    "spectral": [c for c in ("centroid_hz_mean", "centroid_hz_std", "bandwidth_hz_mean", "bandwidth_hz_std", "rolloff_hz_mean", "rolloff_hz_std", "flatness_mean", "flatness_std")],
}


def rank(query: dict, library: pd.DataFrame, output: str | Path, top_k: int = 5):
    """Z-score dùng mean/std thư viện; mỗi nhóm có trọng số bằng nhau.

    Loại cột thiếu >20%, gần hằng hoặc file query không có quan trắc hữu hạn.
    Không đánh giá bản ghi cùng speaker trong benchmark, nếu có speaker_id.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    usable = library[library["status"].eq("ok")].copy()
    # Nếu query vốn nằm trong thư viện, không để nó tự tìm thấy chính nó hay
    # làm scaler học từ chính hàng truy vấn.
    if "path" in usable and query.get("path"):
        query_path = str(Path(query["path"]).resolve()).casefold()
        usable = usable[usable["path"].astype(str).map(lambda p: str(Path(p).resolve()).casefold() != query_path)]
    if len(usable) < 3:
        raise ValueError("Cần ít nhất 3 bản ghi hoạt động ngoài query để lập thư viện tham chiếu.")
    results = usable[["path"]].copy()
    group_distances = []
    group_report = {}
    for group, candidates in GROUPS.items():
        cols = [c for c in candidates if c in usable and c in query]
        if not cols: continue
        arr = usable[cols].apply(pd.to_numeric, errors="coerce")
        coverage = arr.notna().mean()
        cols = list(coverage[coverage >= .8].index)
        if not cols: continue
        mean = arr[cols].mean(); std = arr[cols].std(ddof=0).replace(0, np.nan)
        cols = list(std[std > 1e-8].index)
        if len(cols) < 2: continue
        mean, std = mean[cols], std[cols]
        q = pd.to_numeric(pd.Series(query), errors="coerce").reindex(cols)
        valid = q.notna()
        if int(valid.sum()) < max(2, int(.6*len(cols))): continue
        cols = list(q[valid].index); qz = (q[cols]-mean[cols])/std[cols]
        rz = (arr[cols]-mean[cols])/std[cols]
        # Cột query hữu hạn; hàng candidate phải có đủ 80% cột để được xếp hạng.
        valid_rows = rz.notna().mean(axis=1) >= .8
        d = np.sqrt(((rz-qz)**2).mean(axis=1, skipna=True)).where(valid_rows)
        sim = rz.apply(lambda row: _cosine(row, qz), axis=1)
        results[group+"_distance"] = d
        results[group+"_cosine"] = sim
        group_distances.append(d)
        group_report[group] = {"dimensions": len(cols), "query_overlap_fraction": float(len(cols)/len(candidates)),
                               "closest_median_distance": float(d.median()) if d.notna().any() else None}
    if not group_distances: raise ValueError("Không có đủ đặc trưng chung để so sánh query và thư viện.")
    results["distance"] = pd.concat(group_distances, axis=1).mean(axis=1, skipna=True)
    results = results.dropna(subset=["distance"]).sort_values("distance").head(top_k)
    results.to_csv(output/"search_results.csv", index=False)
    (output/"search_report.json").write_text(json.dumps({"method": "balanced mean group z-distance; reference fitted on valid corpus rows", "groups": group_report, "top_k": top_k}, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def _cosine(a, b):
    valid = a.notna() & b.notna()
    if valid.sum() < 2: return np.nan
    av, bv = a[valid].to_numpy(), b[valid].to_numpy()
    den = np.linalg.norm(av)*np.linalg.norm(bv)
    return float(np.dot(av,bv)/den) if den > 1e-12 else np.nan


def compare_two(a: dict, b: dict, output: str | Path):
    """So sánh cặp sau chuẩn hoá từ mean/std gộp của thư viện tham chiếu."""
    ref_path = Path(a["_reference_csv"])
    library = pd.read_csv(ref_path)
    if "path" in library:
        excluded = {str(Path(a["path"]).resolve()).casefold(), str(Path(b["path"]).resolve()).casefold()}
        library = library[~library["path"].astype(str).map(lambda p: str(Path(p).resolve()).casefold() in excluded)]
    if len(library) < 3:
        raise ValueError("Cần ít nhất ba hàng tham chiếu sau khi loại hai tệp đang so sánh.")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    report = {}
    for group, candidates in GROUPS.items():
        cols = [c for c in candidates if c in library and c in a and c in b]
        if not cols: continue
        frame = library[cols].apply(pd.to_numeric, errors="coerce")
        cols = list((frame.notna().mean() >= .8).index[(frame.notna().mean() >= .8)])
        if len(cols)<2: continue
        mean, std = frame[cols].mean(), frame[cols].std(ddof=0).replace(0,np.nan)
        valid = [c for c in cols if np.isfinite(a[c]) and np.isfinite(b[c]) and std[c] > 1e-8]
        if len(valid)<2: continue
        za = (np.array([a[c] for c in valid])-mean[valid].to_numpy())/std[valid].to_numpy()
        zb = (np.array([b[c] for c in valid])-mean[valid].to_numpy())/std[valid].to_numpy()
        den=np.linalg.norm(za)*np.linalg.norm(zb)
        report[group]={"dimensions":len(valid), "euclidean_distance":float(np.linalg.norm(za-zb)),
                       "cosine_similarity":float(np.dot(za,zb)/den) if den>1e-12 else None}
    most_different = max(report, key=lambda key: report[key]["euclidean_distance"]) if report else None
    (output/"pair_comparison.json").write_text(json.dumps({"groups":report,"largest_normalized_difference_group":most_different,"warning":"Độ tương đồng đặc trưng không xác minh danh tính người nói."}, ensure_ascii=False, indent=2),encoding="utf-8")
    return report
