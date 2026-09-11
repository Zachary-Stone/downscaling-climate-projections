from pathlib import Path

import torch
from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    @computed_field
    @property
    def PROJECT_ROOT(self) -> Path:
        """Absolute repository root: the directory that contains ``src``."""
        return Path(__file__).resolve().parent.parent

    @computed_field
    @property
    def DEVICE(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOCAL_DATA_CANDIDATES: list[Path] = [
        Path("data-tutorial/NZ_domain"),
        Path("NZ_domain"),
    ]

    # Driving GCMs
    GCM_TRAIN: str = "ACCESS-CM2"  # used for training + projection
    GCM_TRANSFER: str = (
        "EC-Earth3"  # a second, independent GCM used only for projection
    )

    # Time periods
    TRAIN_PERIOD: str = "1961-1980"  # model is trained here (present-day climate)
    HIST_PERIOD: str = (
        "1981-2000"  # independent historical test + present-day reference
    )
    FUTURE_PERIOD: str = "2080-2099"  # end-of-century projection
    VAL_SPLIT_YEAR: int = 1977  # validation period for training

    # Predictors
    # 15 large-scale fields: u, v, q, t, z at 850/700/500 hPa.
    # 🔧 You can drop levels/variables to see how skill changes
    PRED_VARS: list[str] = [
        f"{v}_{lev}" for v in ["u", "v", "q", "t", "z"] for lev in [850, 700, 500]
    ]

    # Model / training hyper-parameters
    FILTERS_LAST_CONV: int = 1  # 🔧 channels in the last conv layer (model capacity)
    BATCH_SIZE: int = 32  # 🔧
    LEARNING_RATE: float = 1e-4  # 🔧
    NUM_EPOCHS: int = 50  # 🔧 increase for longer training

    # Where trained models are saved (created if needed)
    MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"

    @computed_field
    @property
    def MODEL_PATH(self) -> Path:
        return self.MODELS_DIR / "deepesd_pr_nz.pt"

    # CodeCarbon output (created if needed)
    CODECARBON_DIR: Path = Path(__file__).resolve().parent.parent / "code_carbon"

    @field_validator("MODELS_DIR", "CODECARBON_DIR")
    @classmethod
    def create_output_directory(cls, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    # Precipitation settings
    WET_DAY_THRESHOLD: float = 1.0  # mm/day, threshold used to define a "wet day"


settings = Settings()
