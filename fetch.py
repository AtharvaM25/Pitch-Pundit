import requests
import json
from pathlib import Path


def fetch_worldcup(year, force=False):
    URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/{year}/worldcup.json"

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"worldcup_{year}.json"

    if path.exists() and not force:
        with open(path, encoding="utf-8") as f:
            return (json.load(f))

    response = requests.get(URL.format(year=year), timeout=10)
    response.raise_for_status()
    data = response.json()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


if __name__ == "__main__":
    fetch_worldcup(2026, force=True)
    fetch_worldcup(2022)
    fetch_worldcup(2018)
    print("Done")
