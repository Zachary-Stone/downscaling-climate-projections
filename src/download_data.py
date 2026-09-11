import zipfile
from pathlib import Path

import requests

from config import settings

# Zenodo: tutorial subset (~3.3 GB). Full CORDEX-ML-Bench: https://doi.org/10.5281/zenodo.20985924
ZENODO_RECORD: str = "21425433"
ZENODO_ZIP_NAME: str = "NZ-subset.zip"
ZENODO_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD}/files/{ZENODO_ZIP_NAME}?download=1"
)
GCM_TRANSFER: str = "EC-Earth3"
REQUIRED_RELATIVE_PATHS = [
    Path("train/ESD_pseudo_reality/predictors")
    / f"{settings.GCM_TRAIN}_{settings.TRAIN_PERIOD}.nc",
    Path("train/ESD_pseudo_reality/target")
    / f"pr_tasmax_{settings.GCM_TRAIN}_{settings.TRAIN_PERIOD}.nc",
    Path("test/historical/predictors/perfect")
    / f"{settings.GCM_TRAIN}_{settings.HIST_PERIOD}.nc",
    Path("test/historical/predictors/imperfect")
    / f"{settings.GCM_TRAIN}_{settings.HIST_PERIOD}.nc",
    Path("test/historical/predictors/imperfect")
    / f"{GCM_TRANSFER}_{settings.HIST_PERIOD}.nc",
    Path("test/historical/target")
    / f"pr_tasmax_{settings.GCM_TRAIN}_{settings.HIST_PERIOD}.nc",
    Path("test/historical/target")
    / f"pr_tasmax_{GCM_TRANSFER}_{settings.HIST_PERIOD}.nc",
    Path("test/end_century/predictors/imperfect")
    / f"{settings.GCM_TRAIN}_{settings.FUTURE_PERIOD}.nc",
    Path("test/end_century/predictors/imperfect")
    / f"{GCM_TRANSFER}_{settings.FUTURE_PERIOD}.nc",
    Path("test/end_century/target")
    / f"pr_tasmax_{settings.GCM_TRAIN}_{settings.FUTURE_PERIOD}.nc",
    Path("test/end_century/target")
    / f"pr_tasmax_{GCM_TRANSFER}_{settings.FUTURE_PERIOD}.nc",
]


def _has_required_files(data_root: Path) -> bool:
    """
    Check whether a data directory contains every tutorial input file.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory expected to contain the NZ tutorial data layout.

    Returns
    -------
    bool
        True when every path in ``REQUIRED_RELATIVE_PATHS`` exists beneath
        ``data_root``; otherwise, False.
    """
    return all((data_root / rel).exists() for rel in REQUIRED_RELATIVE_PATHS)


def download_nz_domain(
    extract_to: Path = Path("."), zip_path: str | Path | None = None
) -> Path:
    """
    Download and extract the tutorial NZ data subset from Zenodo.

    Parameters
    ----------
    extract_to : pathlib.Path, optional
        Directory into which the archive is extracted. Default is the current
        working directory.
    zip_path : str or pathlib.Path, optional
        Temporary archive path. When omitted, uses ``NZ-subset.zip`` within
        ``extract_to``.

    Returns
    -------
    pathlib.Path
        Extracted ``NZ_domain`` directory.

    Raises
    ------
    requests.RequestException
        If the archive cannot be downloaded successfully.
    zipfile.BadZipFile
        If the downloaded archive is not a valid ZIP file.
    FileNotFoundError
        If extraction does not produce the expected ``NZ_domain`` directory.
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    zip_path = Path(zip_path) if zip_path is not None else extract_to / ZENODO_ZIP_NAME

    print(f"Downloading {ZENODO_ZIP_NAME} from Zenodo record {ZENODO_RECORD} ...")
    print(f"URL: {ZENODO_URL}")
    with requests.get(ZENODO_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(zip_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100.0 * downloaded / total
                    print(
                        f"\r  {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
        if total:
            print()

    print(f"Extracting to {extract_to.resolve()} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    zip_path.unlink(missing_ok=True)

    nz_domain_dir = extract_to / "NZ_domain"
    if not nz_domain_dir.exists():
        raise FileNotFoundError(
            f"Expected extracted folder '{nz_domain_dir}' after download, but it was not found."
        )
    return nz_domain_dir


def resolve_data_root() -> Path:
    """
    Locate an existing NZ dataset or download it into the project data directory.

    Returns
    -------
    pathlib.Path
        Directory containing the complete NZ tutorial data layout.

    Raises
    ------
    FileNotFoundError
        If the downloaded data does not contain every required input file.
    requests.RequestException
        If data must be downloaded and the request fails.
    zipfile.BadZipFile
        If data must be downloaded and the received archive is invalid.
    """
    for candidate in settings.LOCAL_DATA_CANDIDATES:
        if not candidate.is_absolute():
            candidate = settings.PROJECT_ROOT / candidate
        if _has_required_files(candidate):
            print(f"Using local data at {candidate.resolve()}")
            return candidate

    # Common layout after a previous Zenodo extract in the working directory
    zenodo_root = settings.PROJECT_ROOT / "data" / "NZ_domain"
    if _has_required_files(zenodo_root):
        print(f"Using previously downloaded data at {zenodo_root.resolve()}")
        return zenodo_root

    data_dir = settings.PROJECT_ROOT / "data"
    nz_domain_dir = download_nz_domain(extract_to=data_dir)

    if not _has_required_files(nz_domain_dir):
        missing = [
            str(rel)
            for rel in REQUIRED_RELATIVE_PATHS
            if not (nz_domain_dir / rel).exists()
        ]
        raise FileNotFoundError(
            "Download finished but required files are missing:\n  - "
            + "\n  - ".join(missing)
        )
    print(f"DATA_ROOT ready at {nz_domain_dir.resolve()}")
    return nz_domain_dir


if __name__ == "__main__":
    DATA_ROOT = resolve_data_root()
    print("DATA_ROOT =", DATA_ROOT.resolve())
    print("Required files present:", _has_required_files(DATA_ROOT))
