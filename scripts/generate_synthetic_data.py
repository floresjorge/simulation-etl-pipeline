"""
Generate synthetic DogDog sales data.

CHURN_CUTOFF_DATE is the single source of truth for the churn label.
transform.py and churn_model.py import this constant — never redefine it.
"""
import calendar
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ── Churn definition ─────────────────────────────────────────────────────────
CHURN_CUTOFF_DATE = date(2024, 6, 1)   # customers with no visit on/after this date → churned

# ── Dataset window ────────────────────────────────────────────────────────────
DATASET_START = date(2023, 1, 1)
DATASET_END   = date(2024, 6, 30)

# ── Population ────────────────────────────────────────────────────────────────
NUM_CUSTOMERS = 320
NUM_EMPLOYEES = 8
CHURN_RATE    = 0.25
BASE_LAMBDA   = 1.85   # expected visits per customer per month (before seasonality)

# ── Business domain ───────────────────────────────────────────────────────────
SERVICES = ["Guardería", "Baño", "Corte", "Hotel", "Adiestramiento"]
SERVICE_WEIGHTS = [0.40, 0.25, 0.15, 0.12, 0.08]
SERVICE_PRICE_RANGES = {
    "Guardería":     (150, 250),
    "Baño":          (200, 450),
    "Corte":         (250, 500),
    "Hotel":         (350, 600),
    "Adiestramiento":(400, 800),
}
PAYMENT_METHODS = ["Efectivo", "Tarjeta", "Transferencia"]
PAYMENT_WEIGHTS = [0.45, 0.40, 0.15]
NOTES_POOL = [
    "cliente frecuente",
    "perro agresivo",
    "descuento aplicado",
    "primera visita",
    "pago pendiente",
    "alérgico a champú",
    "requiere cita previa",
    "cliente VIP",
    "traer vacunas",
    "propina incluida",
]
# December +60%, February -30%; all other months get uniform noise in [0.85, 1.15]
MONTH_MULTIPLIERS = {2: 0.70, 12: 1.60}

DOG_NAMES = [
    "Max", "Luna", "Buddy", "Bella", "Rocky", "Coco", "Toby", "Daisy",
    "Charlie", "Lola", "Duke", "Mia", "Bear", "Nala", "Zeus", "Kira",
    "Rex", "Nena", "Thor", "Canela", "Simba", "Frida", "Boby", "Maya",
    "Sultan", "Paloma", "Lobo", "Bonita", "Sammy", "Estrella", "Tito", "Panda",
    "Gus", "Chispa", "Ringo", "Manchas", "Negro", "Blanca", "Tigrillo", "Pepa",
    "Chico", "Nina", "Pancho", "Rosa", "Beto", "Sasha", "Gordo", "Princesa",
    "Chato", "Fifi",
]

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "sales_export.csv"


def _month_multiplier(month: int, rng: np.random.Generator) -> float:
    if month in MONTH_MULTIPLIERS:
        return MONTH_MULTIPLIERS[month]
    return float(rng.uniform(0.85, 1.15))


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Faker.seed(seed)

    num_churned = round(NUM_CUSTOMERS * CHURN_RATE)
    customer_ids = [f"C-{str(i + 1).zfill(3)}" for i in range(NUM_CUSTOMERS)]
    churned_set  = set(customer_ids[:num_churned])

    # One dog per customer
    name_pool = (DOG_NAMES * (NUM_CUSTOMERS // len(DOG_NAMES) + 1))[:NUM_CUSTOMERS]
    shuffled = name_pool.copy()
    rng.shuffle(shuffled)
    dog_map = dict(zip(customer_ids, shuffled))

    # Slight service preference per customer
    pref_service = {
        cid: SERVICES[int(rng.choice(len(SERVICES), p=SERVICE_WEIGHTS))]
        for cid in customer_ids
    }

    rows = []
    sale_counter = 1
    periods = pd.period_range(
        start=f"{DATASET_START.year}-{DATASET_START.month:02d}",
        end=f"{DATASET_END.year}-{DATASET_END.month:02d}",
        freq="M",
    )

    for period in periods:
        year, month = period.year, period.month
        mult = _month_multiplier(month, rng)
        _, days_in_month = calendar.monthrange(year, month)
        # Cap last month at DATASET_END
        if year == DATASET_END.year and month == DATASET_END.month:
            days_in_month = DATASET_END.day

        for cid in customer_ids:
            is_churned = cid in churned_set
            # Churned customers never visit on/after CHURN_CUTOFF_DATE
            if is_churned and date(year, month, 1) >= CHURN_CUTOFF_DATE:
                continue

            n_visits = int(rng.poisson(BASE_LAMBDA * mult))
            for _ in range(n_visits):
                day = int(rng.integers(1, days_in_month + 1))
                visit_date = date(year, month, day)

                # Guard: churned customers cannot visit on/after cutoff
                if is_churned and visit_date >= CHURN_CUTOFF_DATE:
                    continue

                # Service with customer-level preference bias
                sw = SERVICE_WEIGHTS.copy()
                sw[SERVICES.index(pref_service[cid])] *= 2.0
                total = sum(sw)
                sw = [w / total for w in sw]
                service = SERVICES[int(rng.choice(len(SERVICES), p=sw))]

                lo, hi = SERVICE_PRICE_RANGES[service]
                price = float(round(int(rng.integers(lo, hi + 1)) / 10) * 10)

                payment  = PAYMENT_METHODS[int(rng.choice(len(PAYMENT_METHODS), p=PAYMENT_WEIGHTS))]
                employee = f"EMP-{int(rng.integers(1, NUM_EMPLOYEES + 1)):02d}"
                note     = str(rng.choice(NOTES_POOL)) if rng.random() < 0.60 else None

                rows.append({
                    "sale_id":        f"DD-{str(sale_counter).zfill(6)}",
                    "date":           visit_date.isoformat(),
                    "customer_id":    cid,
                    "dog_name":       dog_map[cid],
                    "service":        service,
                    "price_mxn":      price,
                    "payment_method": payment,
                    "employee_id":    employee,
                    "notes":          note,
                    "is_churned":     int(is_churned),
                })
                sale_counter += 1

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    churned_customers = df.groupby("customer_id")["is_churned"].first().sum()
    print(f"Generated {len(df):,} rows → {OUTPUT_PATH}")
    print(f"Customers: {df['customer_id'].nunique()} total, {churned_customers} churned")


if __name__ == "__main__":
    main()
