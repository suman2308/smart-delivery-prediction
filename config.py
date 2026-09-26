"""Configuration settings for CourierAI platform."""
import os
import secrets as _secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.environ.get("DELIVERY_DB_PATH", os.path.join(BASE_DIR, "delivery.db"))
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")


def _load_or_create_secret() -> str:
    """Return a stable per-install session-signing secret.

    Precedence: the SECRET_KEY env var -> a persisted file
    (``instance/secret_key``, created on first use) -> a fresh per-process
    key if the filesystem is read-only.

    A public "dev default" is deliberately NEVER used: the previous
    hardcoded fallback meant any deployment that skipped SECRET_KEY could
    have its session cookies forged (admin takeover / user impersonation).
    """
    env = os.environ.get("SECRET_KEY")
    if env:
        return env
    secret_path = os.path.join(BASE_DIR, "instance", "secret_key")
    try:
        if os.path.isfile(secret_path):
            with open(secret_path, "r", encoding="utf-8") as fh:
                value = fh.read().strip()
            if value:
                return value
        os.makedirs(os.path.dirname(secret_path), exist_ok=True)
        value = _secrets.token_hex(32)
        # Atomic write so concurrent workers never see a half-written file.
        tmp = secret_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(value)
        os.replace(tmp, secret_path)
        try:
            os.chmod(secret_path, 0o600)  # owner-only on Unix (no-op on Windows)
        except OSError:
            pass
        return value
    except OSError:
        # Read-only filesystem: per-process random key. Sessions/keys don't
        # survive restarts, but they can never be forged from a known default.
        return _secrets.token_hex(32)


# Session signing key. Set SECRET_KEY in production (Render/Docker env);
# otherwise a random per-install key is generated and persisted once.
SECRET_KEY = _load_or_create_secret()

# REST API rate limiting (per API key). Overridable for tests / tuning.
API_RATE_LIMIT = os.environ.get("API_RATE_LIMIT", "30 per minute")
API_RATE_LIMIT_DAILY = os.environ.get("API_RATE_LIMIT_DAILY", "1000 per day")
LOGIN_RATE_LIMIT = os.environ.get("LOGIN_RATE_LIMIT", "10 per minute")
REGISTER_RATE_LIMIT = os.environ.get("REGISTER_RATE_LIMIT", "5 per minute")
REGENERATE_KEY_RATE_LIMIT = os.environ.get("REGENERATE_KEY_RATE_LIMIT", "3 per hour")
# Rate-limit storage: Redis in production (REDIS_URL), in-memory otherwise.
# NOTE: in-memory storage is per-process — with multiple gunicorn workers the
# effective budget multiplies by the worker count, so use Redis when scaling.
RATE_LIMIT_STORAGE_URI = (
    os.environ.get("RATE_LIMIT_STORAGE_URI") or os.environ.get("REDIS_URL") or "memory://"
)

# ---------------------------------------------------------------------------
# Input-validation bounds (defence against DoS / DB bloat via crafted input)
# ---------------------------------------------------------------------------
MAX_TEXT_LEN = 64          # max chars for origin / destination / mode / etc.
MAX_TOTAL_PIECES = 10000   # max pieces per consignment
MAX_WEIGHT = 1_000_000.0   # max actual / volumetric / chargeable weight (kg)

# ---------------------------------------------------------------------------
# Subscription plans & prediction quotas
# ---------------------------------------------------------------------------
# Every new account starts on the free "basic" plan with a monthly prediction
# quota. Pro plans are unlimited (demo billing — no real payment processed).
FREE_PLAN_LIMIT = int(os.environ.get("FREE_PLAN_LIMIT", "50"))

PLANS = {
    # limit is resolved dynamically via plan_limit() so FREE_PLAN_LIMIT can be
    # tuned per-environment (and per-test) without rebuilding the dict.
    "basic": {"name": "Free", "price": 0, "period": None, "limit": None},
    "pro_monthly": {"name": "Pro", "price": 299, "period": "month", "limit": None},
    "pro_yearly": {"name": "Pro+", "price": 1299, "period": "year", "limit": None},
}

# Standalone admin panel login (via /admin-login). Any user can open the
# Admin login page; these credentials unlock the panel without needing a
# registered account. Override via env vars in production — the defaults are
# demo credentials and MUST be changed for any public deployment.
ADMIN_LOGIN_EMAIL = os.environ.get("ADMIN_LOGIN_EMAIL", "admin@gmail.com")
ADMIN_LOGIN_PASSWORD = os.environ.get("ADMIN_LOGIN_PASSWORD", "00000000")

# How long each paid plan lasts before it expires back to the free plan.
PLAN_DURATION_DAYS = {"pro_monthly": 30, "pro_yearly": 365}


def plan_limit(plan: str) -> int | None:
    """Monthly prediction limit for a plan. None means unlimited."""
    if plan == "basic":
        return FREE_PLAN_LIMIT
    return PLANS.get(plan, {}).get("limit")
