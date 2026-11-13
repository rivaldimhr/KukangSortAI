"""Run the real models on the sample photos and build the static demo site.

Used by GitHub Actions (.github/workflows/demo.yml). Locally, with the backend
requirements installed:

    python scripts/build_demo.py

Output: site/index.html + site/demo/<photo>.JPG + site/demo/results.json.
The results are exactly what the API returns (/api/sort and /api/behavior),
so the online demo shows real model output, not simulated numbers.
"""
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "kukangsort-backend" / "backend"
SAMPLES = BACKEND / "sample_testing"
OUT = ROOT / "site"
# Left out of the public demo because a person is visible in the frame.
EXCLUDE = {"Copy of SUNP0571.JPG"}

sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402


def clean_name(path: Path) -> str:
    return path.name.replace("Copy of ", "")


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "demo").mkdir(parents=True)

    photos = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.name not in EXCLUDE)
    results = {}
    with TestClient(main.app) as client:  # runs the startup hook, i.e. loads the models
        health = client.get("/health").json()
        if not all(health["models"].values()):
            raise SystemExit(f"Not all models loaded: {health['models']}")
        for photo in photos:
            name = clean_name(photo)
            data = photo.read_bytes()
            files = {"file": (name, data, "image/jpeg")}
            sort = client.post("/api/sort", files=files)
            behavior = client.post("/api/behavior", files={"file": (name, data, "image/jpeg")})
            sort.raise_for_status()
            behavior.raise_for_status()
            results[name] = {"size": len(data), "sort": sort.json(), "behavior": behavior.json()}
            shutil.copyfile(photo, OUT / "demo" / name)
            s = results[name]["sort"]
            print(f"{name}: sort={s['kelas']} ({max(s['prob'].values()):.2f}) "
                  f"behavior={results[name]['behavior'].get('kelas')}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": health["models"],
        "threshold": health.get("threshold"),
        "files": results,
    }
    (OUT / "demo" / "results.json").write_text(json.dumps(payload, indent=1))
    shutil.copyfile(ROOT / "kukangsort-ai.html", OUT / "index.html")
    (OUT / ".nojekyll").touch()
    print(f"Demo built with {len(results)} photos -> {OUT}")


if __name__ == "__main__":
    build()
