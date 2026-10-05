import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
RAW_PATH = Path("data/raw/credit.csv")


def download(path: Path = RAW_PATH) -> Path:
    if path.exists():
        print(f"{path} already exists, skip download")
        return path
    for attempt in range(1, 6):
        try:
            resp = requests.get(URL, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"attempt {attempt} failed: {e}")
            if attempt == 5:
                raise
            time.sleep(5 * attempt)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xls_name = next(n for n in zf.namelist() if n.endswith(".xls"))
        df = pd.read_excel(zf.open(xls_name), header=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved {df.shape} to {path}")
    return path


if __name__ == "__main__":
    download(Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_PATH)
