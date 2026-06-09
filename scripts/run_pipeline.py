"""
Orchestrate the E→T→L pipeline.

Usage:
    python -m scripts.run_pipeline [--env dev|prod]

dev  → SQLite at data/simulation_dev.db, DEBUG logging to stdout
prod → SQLite at data/simulation_prod.db, WARNING+ logging to file
"""
import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.extract import extract
from scripts.transform import transform
from scripts.load import load
from scripts.generate_synthetic_data import OUTPUT_PATH as DEFAULT_CSV_PATH

DATA_DIR = _ROOT / "data"
DB_PATHS = {
    "dev":  DATA_DIR / "simulation_dev.db",
    "prod": DATA_DIR / "simulation_prod.db",
}


def _setup_logging(env: str) -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s — %(message)s"
    if env == "dev":
        logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=[logging.StreamHandler(sys.stdout)])
    else:
        log_file = DATA_DIR / "simulation_pipeline_prod.log"
        logging.basicConfig(level=logging.WARNING, format=fmt, handlers=[logging.FileHandler(log_file)])


def run(env: str = "dev", csv_path: Path = DEFAULT_CSV_PATH) -> None:
    db_path = DB_PATHS[env]
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline] env={env}  source={csv_path.name}  db={db_path.name}")

    raw_df = extract(csv_path)
    print(f"[pipeline] extracted       {len(raw_df):>7,} rows")

    sales_df, customer_df = transform(raw_df)
    print(f"[pipeline] transformed  →  {len(sales_df):>7,} sales rows, {len(customer_df):>3} customer feature rows")

    load(sales_df, customer_df, db_path)
    print(f"[pipeline] loaded      →   {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulation ETL pipeline")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev",
                        help="dev: verbose console logging; prod: WARNING+ to file")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH,
                        help="Path to raw sales CSV")
    args = parser.parse_args()

    _setup_logging(args.env)
    run(env=args.env, csv_path=args.csv)


if __name__ == "__main__":
    main()
