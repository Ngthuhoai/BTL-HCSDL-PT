"""CLI: trích rút batch, trực quan hóa, so sánh cặp và tìm top-k."""
import argparse
import json
import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from audio import load_audio
from features import extract
from visualize import dashboard
from compare import rank, compare_two


def _analyze(path: Path, output: Path, plots=False):
    y, meta = load_audio(path)
    if meta["status"] != "ok":
        return {**meta, "error": "Không đủ hoạt động âm thanh theo ngưỡng năng lượng; không đưa vào so khớp."}
    row, seq = extract(y, meta)
    if plots:
        folder=output/"plots"/path.stem
        dashboard(y,seq,meta,folder)
        sequence_dir=output/"sequences"
        sequence_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(sequence_dir/(path.stem+".npz"),**seq)
    return row


def _files(path):
    if path.is_file(): return [path]
    if not path.is_dir(): raise FileNotFoundError(path)
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3", ".m4a"})


def batch(source: Path, out: Path, plots=False, metadata_path=None):
    files=_files(source)
    if not files: raise ValueError(f"Không tìm thấy audio trong {source}")
    (out/"sequences").mkdir(parents=True, exist_ok=True)
    rows=[]
    for i,p in enumerate(files,1):
        try: rows.append(_analyze(p,out,plots))
        except Exception as exc: rows.append({"path":str(p.resolve()),"status":"error","error":f"{type(exc).__name__}: {exc}"})
        if i % 25 == 0 or i == len(files):
            pd.DataFrame(rows).to_csv(out/"features.csv",index=False)
            print(f"[{i}/{len(files)}] {p.name}",flush=True)
    if metadata_path:
        m=pd.read_csv(metadata_path)
        df=pd.DataFrame(rows)
        if "path" in df and "relative_path" in m:
            df["relative_path"]=df["path"].map(lambda s: None if not isinstance(s,str) else Path(s).name)
            m["relative_path"]=m["relative_path"].map(lambda s:Path(s).name)
            df=df.merge(m.drop(columns=["file_id"],errors="ignore"),on="relative_path",how="left",suffixes=("","_metadata"))
        df.to_csv(out/"features.csv",index=False)
    else:
        df = pd.DataFrame(rows)
    # CSDL SQLite có thể truy vấn trực tiếp; replace một lần sau batch để I/O tuyến tính.
    db = out/"features.sqlite"
    with sqlite3.connect(db) as conn:
        df.to_sql("features", conn, if_exists="replace", index=False)
        if "path" in df:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_features_path ON features(path)")
        if "status" in df:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_features_status ON features(status)")
        conn.commit()
    (out/"analysis_config.json").write_text(json.dumps({"sample_rate_hz":16000,"n_fft":1024,"hop_length":160,"frame_ms":64,"hop_ms":10,"method":"librosa DSP, Praat autocorrelation F0, Praat Burg formants","status_counts":df["status"].value_counts().to_dict() if "status" in df else {}},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Đã lưu {len(rows)} hàng tại {out/'features.csv'}")


def main():
    # Giữ thông báo tiếng Việt hoạt động trên PowerShell có code page cũ.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser=argparse.ArgumentParser(description="DSP truyền thống: phân tích và tìm kiếm giọng nói tương đồng")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("analyze",help="Phân tích WAV hoặc toàn bộ thư mục")
    p.add_argument("source",type=Path);p.add_argument("--output",type=Path,default=Path("output"));p.add_argument("--plots",action="store_true");p.add_argument("--metadata",type=Path)
    p=sub.add_parser("compare",help="So sánh hai bản ghi; cần CSV thư viện để chuẩn hóa")
    p.add_argument("first",type=Path);p.add_argument("second",type=Path);p.add_argument("--reference",type=Path,required=True);p.add_argument("--output",type=Path,default=Path("output/compare"))
    p=sub.add_parser("search",help="Tìm bản ghi gần nhất trong CSV đặc trưng")
    p.add_argument("query",type=Path);p.add_argument("--library",type=Path,required=True);p.add_argument("--output",type=Path,default=Path("output/search"));p.add_argument("--top",type=int,default=5)
    a=parser.parse_args()
    if a.command=="analyze": batch(a.source.resolve(),a.output.resolve(),a.plots,a.metadata.resolve() if a.metadata else None)
    elif a.command=="search":
        q=_analyze(a.query.resolve(),a.output.resolve(),False);lib=pd.read_csv(a.library);result=rank(q,lib,a.output,a.top);print(result.to_string(index=False))
    else:
        a.output.resolve().mkdir(parents=True,exist_ok=True)
        ref=pd.read_csv(a.reference); rows=[]
        for pth in (a.first,a.second): rows.append(_analyze(pth.resolve(),a.output.resolve(),True))
        # CSV reference is persisted for group-wise train-set normalization.
        ref.to_csv(a.output.resolve()/"_reference.csv",index=False)
        for row in rows: row["_reference_csv"]=str((a.output.resolve()/"_reference.csv"))
        print(json.dumps(compare_two(*rows,a.output.resolve()),ensure_ascii=False,indent=2))


if __name__=="__main__":
    try: main()
    except (ValueError, FileNotFoundError) as exc: print(f"Lỗi: {exc}",file=sys.stderr);sys.exit(2)
