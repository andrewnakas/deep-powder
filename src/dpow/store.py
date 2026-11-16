"""Feature stores (read-only, harvested by the two sibling entries) and this
repo's own releases, all GitHub Release assets.

- open-powder-ml `store`: daily deterministic archive aggregates.
- open-powder-ensemble `ens-store*` and `snowpack`: ensemble round features
  and SNOTEL snowpack readings.
- own `models`: champion bundles (`<name>.pkl`) + `<name>.json` metrics.
- own `table`: the assembled training table, one parquet per league and season.

deep-powder harvests nothing itself: no Open-Meteo calls, no archive reads.

Locally everything lives under work/store and work/models; `gh` syncs them
(GH_TOKEN in Actions; the repo is the current GITHUB_REPOSITORY).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

from . import WORK

STORE = WORK / "store"  # deterministic archive features (open-powder-ml's public `store` release)
DET_REPO = os.environ.get("DPOW_DET_REPO", "andrewnakas/open-powder-ml")
ENS_REPO = os.environ.get("DPOW_ENS_REPO", "andrewnakas/open-powder-ensemble")  # ens-store*, snowpack
TABLE = WORK / "table"
MODELS = WORK / "models"


def _repo() -> str | None:
    return os.environ.get("DPOW_REPO") or os.environ.get("GITHUB_REPOSITORY")


def _gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True)


def unit_name(league: str, start, end, ids: list[str]) -> str:
    h = hashlib.sha1(",".join(sorted(ids)).encode()).hexdigest()[:10]
    return f"daily__{league}__{start}__{end}__{h}.parquet"


def _ensure_release(tag: str) -> None:
    repo = _repo()
    if repo and _gh("release", "view", tag, "-R", repo).returncode != 0:
        _gh("release", "create", tag, "-R", repo, "--title", tag, "--notes",
            f"Automated {tag} assets (managed by deep-powder workflows).", "--prerelease")


def remote_assets(tag: str, repo: str | None = None) -> set[str]:
    repo = repo or _repo()
    if not repo:
        return set()
    r = _gh("release", "view", tag, "-R", repo, "--json", "assets", "-q", ".assets[].name")
    return set(r.stdout.split()) if r.returncode == 0 else set()


# The ensemble archive is a family of releases in the sibling repo (GitHub
# allows 1000 assets per release): `ens-store` plus `ens-store-<year>`.
ENS_TAG = "ens-store"


def ens_tags() -> list[str]:
    """Every release of the family that exists, the original first."""
    r = _gh("release", "list", "-R", ENS_REPO, "-L", "200", "--json", "tagName", "-q", ".[].tagName")
    if r.returncode != 0:
        raise RuntimeError(f"could not list releases: {r.stderr[:300]}")
    return sorted(t for t in r.stdout.split() if t == ENS_TAG or t.startswith(ENS_TAG + "-"))


def pull_ens(dest: Path) -> None:
    for tag in ens_tags():
        pull(tag, dest, repo=ENS_REPO)


def pull(tag: str, dest: Path, pattern: str = "*", repo: str | None = None, overwrite: bool = True) -> None:
    """Download a release's assets. One API call lists them; the files come
    from the public github.com/.../releases/download/ links, which don't
    count against the API rate limit (hundreds of per-asset API downloads per
    training job exhausted it)."""
    import fnmatch
    import json
    from concurrent.futures import ThreadPoolExecutor

    import requests

    repo = repo or _repo()
    if not repo:
        return
    dest.mkdir(parents=True, exist_ok=True)
    r = _gh("release", "view", tag, "-R", repo, "--json", "assets")
    if r.returncode != 0:
        return
    assets = [a for a in json.loads(r.stdout)["assets"] if fnmatch.fnmatch(a["name"], pattern)]

    def get(a):
        out = dest / a["name"]
        if out.exists() and (not overwrite or out.stat().st_size == a.get("size", -1)):
            return
        url = f"https://github.com/{repo}/releases/download/{tag}/{a['name']}"
        for attempt in range(5):
            try:
                resp = requests.get(url, timeout=300)
                resp.raise_for_status()
                out.write_bytes(resp.content)
                return
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
        print(f"  could not download {a['name']}", flush=True)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(get, assets))


def push(tag: str, files: list[Path]) -> None:
    repo = _repo()
    if not repo or not files:
        return
    import time

    _ensure_release(tag)
    for i in range(0, len(files), 20):
        for attempt in range(20):  # GitHub's API rate limit resets hourly: wait it out (~80 min) rather than lose work
            r = _gh("release", "upload", tag, "-R", repo, "--clobber", *map(str, files[i : i + 20]))
            if r.returncode == 0:
                break
            if "rate limit" not in r.stderr.lower() and attempt >= 2:
                raise RuntimeError(f"release upload failed: {r.stderr[:300]}")
            print(f"  upload retry {attempt + 1}: {r.stderr.strip()[:120]}", flush=True)
            time.sleep(min(300, 60 * (attempt + 1)))
        else:
            raise RuntimeError(f"release upload failed after retries: {r.stderr[:300]}")


def load_daily(league: str) -> pd.DataFrame:
    # lead-matched recent archive + the stitched 2022-24 history (stitched=1 rows)
    files = sorted(STORE.glob(f"daily__{league}__*.parquet")) + sorted(STORE.glob(f"hist__{league}__*.parquet"))
    if not files:
        return pd.DataFrame()
    key = ["station_id", "model", "lead", "date"]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.drop_duplicates(key, keep="last")
    # the models' own snowfall, harvested later as `snow__<unit>` files
    sf = [f for kind in ("daily", "hist") for f in sorted(STORE.glob(f"snow__{kind}__{league}__*.parquet"))]
    if sf:
        sn = pd.concat([pd.read_parquet(f) for f in sf], ignore_index=True)
        sn["date"] = pd.to_datetime(sn["date"]).dt.date
        sn = sn.drop_duplicates(key, keep="last").rename(columns={"snow": "_snow"})
        df = df.merge(sn[[*key, "_snow"]], on=key, how="left")
        df["snow"] = df["snow"].fillna(df.pop("_snow"))
    return df
