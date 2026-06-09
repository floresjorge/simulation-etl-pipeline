-- Simulation data warehouse schema
-- Upsert keys: sales.sale_id, customer_features.customer_id

CREATE TABLE IF NOT EXISTS sales (
    sale_id         TEXT    PRIMARY KEY,
    date            TEXT    NOT NULL,
    customer_id     TEXT    NOT NULL,
    dog_name        TEXT    NOT NULL,
    service         TEXT    NOT NULL,
    price_mxn       REAL    NOT NULL,
    payment_method  TEXT    NOT NULL,
    employee_id     TEXT    NOT NULL,
    notes           TEXT,
    is_churned      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sales_customer ON sales (customer_id);
CREATE INDEX IF NOT EXISTS idx_sales_date     ON sales (date);

CREATE TABLE IF NOT EXISTS customer_features (
    customer_id           TEXT    PRIMARY KEY,
    days_since_last_visit INTEGER NOT NULL,
    visit_frequency       REAL    NOT NULL,
    avg_spend             REAL    NOT NULL,
    preferred_service     TEXT    NOT NULL,
    total_revenue         REAL    NOT NULL,
    is_churned            INTEGER NOT NULL DEFAULT 0,
    updated_at            TEXT    DEFAULT (datetime('now'))
)
