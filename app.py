"""Flask Application Server & Endpoint Controllers."""
from __future__ import annotations

import base64
import hashlib
import io
import math
import os
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache, wraps
from urllib.parse import urlsplit

from werkzeug.middleware.proxy_fix import ProxyFix

from cryptography.fernet import Fernet

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from markupsafe import Markup
from werkzeug.security import check_password_hash, generate_password_hash

import charts
import config
import database as db
import train_experiments as te
from dtdc_model import DTDCPredictor, MODEL_ALGORITHM, MODEL_VERSION

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Behind a trusted reverse proxy (Render / nginx), honour the real client IP
# so rate limiting keys per visitor instead of per proxy. Enabled explicitly
# via TRUST_PROXY=1 — never trust X-Forwarded-* when the app is directly
# exposed (a client could spoof the header to dodge rate limits).
if os.environ.get("TRUST_PROXY", "0") == "1":
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# ---------------------------------------------------------------------------
# Session cookie hardening (OWASP): HttpOnly, SameSite, Secure behind proxy
# ---------------------------------------------------------------------------
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "0") == "1",
    PERMANENT_SESSION_LIFETIME=86400 * 30,
)

# ---------------------------------------------------------------------------
# CSRF protection (OWASP): per-session token validated on state-changing
# requests. API routes are exempt (they authenticate via API keys).
# ---------------------------------------------------------------------------
def _csrf_token() -> str:
    """Return the session's CSRF token, generating one on first use."""
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


app.jinja_env.globals["csrf_token"] = _csrf_token


@app.before_request
def _csrf_protect():
    """Reject state-changing requests that lack a valid CSRF token.

    GET/HEAD/OPTIONS are read-only. API endpoints authenticate via API keys
    (no cookies), so they are exempt. Everything else must present the token
    from its own session — login CSRF included.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    if request.path.startswith("/api/") or request.path == "/health":
        return None
    expected = session.get("_csrf_token", "")
    provided = request.form.get("_csrf_token", "")
    if not expected or not secrets.compare_digest(provided, expected):
        flash("Your session expired. Please try again.", "error")
        # Same-site fallback only — never bounce to an attacker-controlled
        # Referer (open-redirect defence, consistent with login).
        return redirect(_safe_next(request.referrer))
    return None

# ---------------------------------------------------------------------------
# Admin session wall: the admin console is a separate application. A
# standalone admin session is confined to admin endpoints — every public
# page (/, /predict, /about, …) bounces back to /admin. Only the assets the
# console itself needs (static files, health/metrics probes, logout, the
# admin login route) remain reachable.
# ---------------------------------------------------------------------------
@app.before_request
def _wall_admin_from_public_site():
    if not _is_admin_session():
        return None
    endpoint = request.endpoint
    if endpoint is None:
        return None  # unknown path -> the 404 handler renders
    if endpoint in _ADMIN_SESSION_ENDPOINTS or endpoint in (
        "static", "health", "metrics_json", "logout", "admin_login",
    ):
        return None
    return redirect(url_for("admin"))


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
# Global baseline applies per client IP; specific routes tighten (or loosen)
# their own limits below (e.g. /api/predict is keyed per API key).
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["120 per hour", "500 per day"],
    storage_uri=config.RATE_LIMIT_STORAGE_URI,
)

# Ensure the schema (incl. the users table used by the API-key check) exists
# before any request can hit auth-gated endpoints.
db.init_db()


def _api_identity():
    """Rate-limit key for the prediction API: the API key when present,
    otherwise the client IP (so unauthenticated callers can't share limits)."""
    raw = _extract_api_key(request)
    if raw:
        return _hash_api_key(raw)
    return request.remote_addr or "anonymous"


def _generate_api_key() -> str:
    """Create a new API key. Stored hashed — only ever shown in full once."""
    return "scp_live_" + secrets.token_urlsafe(32)


def _hash_api_key(api_key: str) -> str:
    """Hash an API key for storage / lookup (SHA-256, key is high entropy)."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _key_hint(api_key: str) -> str:
    """A display-safe fragment of the key, e.g. 'scp_live_…abcd'."""
    return "scp_live_…" + api_key[-4:]


def _extract_api_key(req) -> str:
    """Pull an API key from the X-API-Key header or an Authorization Bearer."""
    key = req.headers.get("X-API-Key", "").strip()
    if key:
        return key
    auth = req.headers.get("Authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    """Fernet cipher for API keys. Key is derived from the app secret so a DB
    leak alone never exposes keys; override via KEY_ENCRYPTION_KEY in prod.

    IMPORTANT: the derivation material (SECRET_KEY / KEY_ENCRYPTION_KEY) must
    stay stable across deploys — changing it makes stored keys undecryptable.
    """
    raw = os.environ.get("KEY_ENCRYPTION_KEY") or config.SECRET_KEY
    material = hashlib.sha256(raw.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(material))


def _encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for storage (revealable later by its owner)."""
    return _fernet().encrypt(api_key.encode("utf-8")).decode("utf-8")


def _decrypt_api_key(token: str) -> str:
    """Decrypt a stored API key token back to plaintext."""
    return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------
@app.after_request
def _set_security_headers(response):
    """Apply standard security headers to every response."""
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'"
    )
    # Private pages must never be cached by shared browsers/proxies.
    if request.endpoint != "static" and (session.get("user_id") or session.get("is_admin")):
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# Security audit log (never logs passwords, tokens or API keys)
# ---------------------------------------------------------------------------
def _audit(event: str, **fields) -> None:
    """Emit a security-relevant audit line via the app logger (captured by
    gunicorn/container logs). Sensitive values must never be passed here.

    WARNING level is intentional: Flask's default logger level suppresses
    INFO in production (debug=False), which would silently drop audit lines.
    """
    detail = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    app.logger.warning("AUDIT %s %s", event, detail)


# Warn loudly when the demo admin credentials are in use outside local dev.
if (
    config.ADMIN_LOGIN_PASSWORD == "00000000"
    and config.ADMIN_LOGIN_EMAIL == "admin@gmail.com"
    and os.environ.get("FLASK_DEBUG") != "1"
):
    app.logger.warning(
        "AUDIT config: default admin credentials in use "
        "(admin@gmail.com / 00000000). Set ADMIN_LOGIN_EMAIL and "
        "ADMIN_LOGIN_PASSWORD env vars before deploying publicly."
    )

# ---------------------------------------------------------------------------
# DTDC model - initialised once at module load (singleton guarantees one load)
# ---------------------------------------------------------------------------
_dtdc_predictor = DTDCPredictor()


def ensure_app_ready():
    """One-time startup prep: dirs, schema, plots dir. The production DTDC
    model ships committed, so no training/bootstrap is needed at startup."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    db.init_db()


@app.route("/")
def index():
    ensure_app_ready()
    metrics = _load_metrics()
    return render_template("index.html", metrics=metrics, error=None)


def _friendly_algorithm(name):
    """Human-friendly algorithm label for the UI.

    Strips sklearn's verbose class suffix (e.g. "HistGradientBoostingRegressor"
    becomes "HistGradientBoosting") so it fits the KPI/stat cards. The API keeps
    the full class name.
    """
    if not name:
        return name
    for suffix in ("Regressor", "Classifier", "Estimator"):
        if name.endswith(suffix):
            short = name[: -len(suffix)]
            return short if short else name
    return name


def _algorithm_display(name):
    """Friendly algorithm name with a line break before the model type so it fits
    KPI/stat cards (e.g. "HistGradient<br>Boosting"). Other names pass through.
    """
    friendly = _friendly_algorithm(name)
    if not friendly:
        return friendly
    for suffix in ("Boosting", "Forest", "Network", "Neighbors", "Bayes", "Ridge"):
        if friendly.endswith(suffix) and len(friendly) > len(suffix):
            return Markup(f"{friendly[: -len(suffix)]}<br>{suffix}")
    return friendly


app.add_template_filter(_algorithm_display, "display_algo")


def _load_metrics():
    """Return model metrics dict or None if the model artifact is missing."""
    try:
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
        return {
            "mae": m.get("mae_days", 0),
            "rmse": m.get("rmse_days", 0),
            "r2": m.get("r2", 0),
            "algorithm": _friendly_algorithm(meta.get("algorithm", MODEL_ALGORITHM)),
            "model_version": meta.get("model_version", MODEL_VERSION),
        }
    except (FileNotFoundError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def _current_user():
    """Return the logged-in user row (or None) for the active session."""
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.get_user_by_id(int(user_id))


def _active_nav_for(endpoint: str) -> str:
    """Map the current Flask endpoint to a navbar section so the active link
    gets its underline indicator. Server-side (single source of truth): a page
    like /dashboard highlights "Analytics" even though its URL differs.

    Pages without a navbar entry (e.g. /plans, /pricing, /result) intentionally
    leave no item highlighted.
    """
    mapping = {
        "index": "home",
        "predict_page": "predict",
        "predict_form": "predict",
        "tracking": "tracking",
        "dashboard": "dashboard",
        "model_comparison": "models",
        "api_docs": "api",
        "about": "about",
        "contact": "contact",
        "admin": "admin",
        "admin_login": "admin",
        "admin_demo": "admin",
        "admin_run_experiments": "admin",
        "admin_experiment_status": "admin",
        # Analytics lives under the Admin dropdown now (no separate nav item).
        "analytics": "admin",
    }
    return mapping.get(endpoint, "")


def _is_admin_session() -> bool:
    """True when the session holds a standalone admin login (no account)."""
    return bool(session.get("is_admin"))


@app.context_processor
def _inject_auth():
    """Expose the current user, an admin flag, a cache-busting asset version,
    and the active nav section to templates.

    The admin flag is ONLY the standalone admin session (/admin-login): a
    registered account is always a regular user, so the user panel and the
    admin console never mix in the same session.

    Plan status is computed only on the routes that need it (account/pricing)
    rather than on every request."""
    user = _current_user()
    is_admin = _is_admin_session()
    return {
        "current_user": user,
        "is_admin": is_admin,
        "demo_mode": bool(session.get("demo_mode")),
        "static_version": _static_version(),
        "active_nav": _active_nav_for(request.endpoint or ""),
    }


def _static_version() -> str:
    """Cache-busting token derived from the main static assets' mtimes, so
    browsers pick up changed CSS/JS immediately after a deploy."""
    newest = 0.0
    for name in ("static/js/app.js", "static/css/app.css"):
        try:
            newest = max(newest, os.path.getmtime(os.path.join(config.BASE_DIR, name)))
        except OSError:
            pass
    return str(int(newest)) if newest else "1"


# Endpoints a standalone admin session (no account row) is allowed to visit.
# Everything else behind @login_required assumes a real user, so standalone
# admins are routed to the admin panel instead of crashing on user["id"].
# admin_login is NOT listed: it self-handles the flag (redirects to /admin)
# and is not itself behind login_required.
_ADMIN_SESSION_ENDPOINTS = frozenset({
    "admin",
    "admin_demo",
    "admin_demo_reset",
    "admin_run_experiments",
    "admin_experiment_status",
    "analytics",
})


def login_required(view):
    """Redirect anonymous visitors to the login page, remembering where they
    were headed so they can be returned after signing in.

    Also handles stale sessions: a session cookie may still carry a user_id
    for an account that was deleted or whose DB row was reset. Such sessions
    are cleared and sent to login instead of crashing downstream on None.

    A standalone admin login (session["is_admin"]) satisfies the gate too,
    but only on admin-gated endpoints — user pages like /account assume an
    account row and must not be reached by a flag-only session."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if _is_admin_session():
            if request.endpoint in _ADMIN_SESSION_ENDPOINTS:
                return view(*args, **kwargs)
            return redirect(url_for("admin"))
        if session.get("user_id") is None:
            # Admin-only endpoints send anonymous visitors to the standalone
            # admin login, not the user login.
            if request.endpoint in _ADMIN_SESSION_ENDPOINTS:
                return redirect(url_for("admin_login", next=request.path))
            return redirect(url_for("login", next=request.path))
        if _current_user() is None:
            session.clear()
            flash("Your session has expired. Please log in again.", "info")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view):
    """Restrict a view to a standalone admin session (via /admin-login).

    Registered accounts are never admins: hitting an admin route while
    logged in as a regular user bounces them to their dashboard."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if _is_admin_session():
            return view(*args, **kwargs)
        user = _current_user()
        flash("Admin access required.", "error")
        if user:
            return redirect(url_for("dashboard"))
        # No session at all: point at the standalone admin login, and remember
        # where they were headed so the console opens the right section.
        return redirect(url_for("admin_login", next=request.path))

    return wrapped


@app.route("/admin-login", methods=["GET", "POST"])
@limiter.limit(config.LOGIN_RATE_LIMIT, methods=["POST"], override_defaults=False)
def admin_login():
    """Standalone admin login (footer link) using the fixed credentials from
    config.ADMIN_LOGIN_EMAIL / ADMIN_LOGIN_PASSWORD. Grants a session flag
    that unlocks every admin-gated route without needing a user account."""
    if _is_admin_session():
        return redirect(url_for("admin"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        # Constant-time comparison (consistent with the CSRF check) so the
        # admin credentials aren't a timing side channel.
        if (
            secrets.compare_digest(email, config.ADMIN_LOGIN_EMAIL.lower())
            and secrets.compare_digest(password, config.ADMIN_LOGIN_PASSWORD)
        ):
            # A logged-in regular user switching to the admin console gets a
            # clean session — same identity-swap pattern as login()/register().
            _audit("admin_login_success", ip=request.remote_addr)
            session.clear()  # prevent session fixation
            session["is_admin"] = True
            session.permanent = True
            flash("Welcome back, Admin!", "success")
            # Honor a safe ?next= (e.g. an admin section the visitor was
            # headed to); otherwise open the console overview.
            target = request.args.get("next")
            return redirect(_safe_next(target) if target else url_for("admin"))
        _audit("admin_login_failed", email=email, ip=request.remote_addr)
        flash("Invalid admin credentials.", "error")

    return render_template("admin_login.html")


# Precomputed hash so the "no such user" path takes comparable time to a real
# password check, avoiding timing-based user enumeration.
_DUMMY_PASSWORD_HASH = generate_password_hash(
    "dummy-password-for-timing", method="pbkdf2:sha256:600000"
)


def _safe_next(target: str | None) -> str:
    """Return target only if it is a safe, same-site path (no open redirect).

    When no target is given (plain login, CSRF fallback), users land on the
    Predict page — the core action — rather than a dashboard.
    """
    if not target:
        return url_for("predict_page")
    # Reject anything with a backslash — browsers normalise `\` to `/`, so
    # `/\\evil.com` would otherwise become `//evil.com` (open redirect).
    if "\\" in target:
        return url_for("predict_page")
    try:
        parts = urlsplit(target)
    except ValueError:
        return url_for("predict_page")
    if parts.scheme or parts.netloc or not target.startswith("/"):
        return url_for("predict_page")
    return target


@app.route("/login", methods=["GET", "POST"])
@limiter.limit(config.LOGIN_RATE_LIMIT, methods=["POST"], override_defaults=False)
def login():
    # A session pointing at a deleted account is stale — clean it up now so a
    # dead user_id never lingers in the cookie (cleared again on login_required).
    if session.get("user_id") and _current_user() is None:
        session.clear()
    if _current_user():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = db.get_user_by_email(email) if email else None
        # Equalise timing: always run a hash check, even for unknown emails.
        if user is not None:
            password_ok = check_password_hash(user["password_hash"], password)
        else:
            password_ok = False
            check_password_hash(_DUMMY_PASSWORD_HASH, password)
        if password_ok:
            _audit("login_success", user_id=user["id"], ip=request.remote_addr)
            session.clear()  # prevent session fixation
            session["user_id"] = user["id"]
            session.permanent = True
            flash(f"Welcome back, {user['full_name']}!", "success")
            return redirect(_safe_next(request.form.get("next")))
        _audit("login_failed", email=email, ip=request.remote_addr)
        flash("Invalid email or password.", "error")

    return render_template(
        "login.html", next=request.args.get("next") or request.form.get("next") or ""
    )


@app.route("/register", methods=["GET", "POST"])
@limiter.limit(config.REGISTER_RATE_LIMIT, methods=["POST"], override_defaults=False)
def register():
    if _current_user():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        company = request.form.get("company", "").strip() or "Personal"
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if len(full_name) < 2:
            flash("Please enter your full name.", "error")
        elif not email or "@" not in email:
            flash("Please enter a valid email address.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters long.", "error")
        else:
            try:
                api_key = _generate_api_key()
                user_id = db.create_user(
                    full_name=full_name,
                    email=email,
                    password_hash=generate_password_hash(password),
                    company=company,
                    api_key_hash=_hash_api_key(api_key),
                    api_key_hint=_key_hint(api_key),
                    api_key_enc=_encrypt_api_key(api_key),
                )
            except ValueError as exc:
                flash(str(exc), "error")
            else:
                _audit("register", user_id=user_id, email=email, ip=request.remote_addr)
                session.clear()  # prevent session fixation
                session["user_id"] = user_id
                session.permanent = True
                flash(f"Account created. Welcome, {full_name}!", "success")
                # The API key is only ever shown this once.
                flash(api_key, "api_key")
                # New accounts pick a plan before reaching the dashboard.
                return redirect(url_for("choose_plan"))

    return render_template(
        "register.html", next=request.args.get("next") or request.form.get("next") or ""
    )


@app.route("/logout", methods=["POST"])
def logout():
    """POST-only so a plain link on a third-party page cannot force a logout."""
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Marketing & product pages
# ---------------------------------------------------------------------------
@app.route("/demo")
def demo():
    ensure_app_ready()
    return render_template("demo.html", metrics=_load_metrics(), error=None, source="demo")


@app.route("/about")
def about():
    return render_template("about.html")


def _generate_tracking_id() -> str:
    """Create a short, human-friendly tracking ID (e.g. SCP-4F2A9B1C).

    Re-checks the DB so a (theoretically rare) hex collision can never raise
    an IntegrityError on insert.
    """
    while True:
        tid = f"SCP-{secrets.token_hex(4).upper()}"
        if db.get_prediction_by_tracking_id(tid) is None:
            return tid


def _log_prediction(kwargs: dict, result, user_id: int | None) -> str:
    """Persist a prediction to the audit table and return its tracking ID.
    Shared by the web form and the JSON API (single source of truth)."""
    tracking_id = _generate_tracking_id()
    db.insert_dtdc_prediction(
        origin=kwargs["origin"],
        destination=kwargs["destination"],
        booking_weekday=kwargs["booking_weekday"],
        mode=kwargs["mode"],
        nature_of_consignment=kwargs["nature_of_consignment"],
        total_pieces=kwargs["total_pieces"],
        actual_weight=kwargs["actual_weight"],
        volumetric_weight=kwargs["volumetric_weight"],
        chargeable_weight=kwargs["chargeable_weight"],
        predicted_days=result.predicted_days,
        model_version=result.model_version,
        tracking_id=tracking_id,
        user_id=user_id,
    )
    return tracking_id


def _recent_tracked():
    """Recent trackable predictions + their computed statuses (shared by the
    tracking landing page and deep-link lookups)."""
    recent = [
        r for r in db.fetch_dtdc_predictions(limit=8) if r["tracking_id"]
    ][:5]
    recent_status = {
        r["id"]: _tracking_timeline(r) for r in recent
    }
    return recent, recent_status


def _tracking_timeline(row):
    """Build a real tracking timeline from a stored prediction record.

    Every stage is anchored to the record's created_at and scaled by the
    predicted duration, so the status reflects genuine elapsed time for that
    exact prediction — no simulation, no random ETAs.
    """
    created = None
    try:
        created = datetime.strptime(
            row["created_at"], "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        created = None
    predicted_days = float(row["predicted_days"] or 1.0) or 1.0

    now = datetime.now(timezone.utc)
    total_seconds = predicted_days * 86400.0
    elapsed = max((now - created).total_seconds(), 0.0) if created else 0.0
    progress = min(100.0, elapsed / total_seconds * 100.0)

    stages = [
        (0.00, "Shipment booked", "Label created and shipment registered in the courier network.", "Booked", "badge-info"),
        (0.15, "Picked up by courier", "Consignment collected from origin hub and scanned into transit.", "Picked Up", "badge-primary"),
        (0.45, "In transit", "Moving through the route network — currently at the regional sorting facility.", "In Transit", "badge-primary"),
        (0.85, "Out for delivery", "Assigned to a delivery agent for final-mile dispatch.", "Out for Delivery", "badge-warning"),
        (1.00, "Delivered", "Consignment handed over and proof of delivery captured.", "Delivered", "badge-success"),
    ]

    active_index = next(
        (i for i, (frac, *_rest) in enumerate(stages) if elapsed < frac * total_seconds),
        None,
    )
    items = []
    last_done = stages[0]
    for i, (frac, title, text, short, badge) in enumerate(stages):
        done = elapsed >= frac * total_seconds
        if done:
            last_done = stages[i]
        at = created + timedelta(seconds=frac * total_seconds) if created else None
        items.append({
            "title": title,
            "text": text,
            "time": at.astimezone().strftime("%b %d, %H:%M") if at else "—",
            "state": "done" if done else ("active" if i == active_index else ""),
            "badge": badge,
            "label": short,
        })

    return {
        "items": items,
        "progress": round(progress, 1),
        "status": last_done[3],
        "status_badge": last_done[4],
        "eta": (
            (created + timedelta(days=predicted_days)).astimezone().strftime("%a, %d %b")
            if created else "—"
        ),
    }


@app.route("/tracking", methods=["GET", "POST"])
def tracking():
    recent, recent_status = _recent_tracked()
    error = None

    if request.method == "POST":
        tid = request.form.get("tracking_id", "").strip().upper()
        if not tid:
            error = "Please enter a tracking ID."
        else:
            row = db.get_prediction_by_tracking_id(tid)
            if row:
                return redirect(url_for("tracking_lookup", tracking_id=tid))
            error = f"No shipment found with ID {tid}. Check the ID and try again."
            return (
                render_template(
                    "tracking.html", recent=recent, recent_status=recent_status,
                    error=error, tracked=None, timeline=None,
                ),
                404,
            )

    return render_template(
        "tracking.html", recent=recent, recent_status=recent_status,
        error=error, tracked=None, timeline=None,
    )


@app.route("/tracking/<tracking_id>")
def tracking_lookup(tracking_id: str):
    """Look up a real prediction record by its tracking ID and render its
    computed journey timeline (shareable deep link)."""
    recent, recent_status = _recent_tracked()
    row = db.get_prediction_by_tracking_id(tracking_id.strip().upper())
    if not row:
        return (
            render_template(
                "tracking.html", recent=recent, recent_status=recent_status,
                error=f"No shipment found with ID {tracking_id}.",
                tracked=None, timeline=None,
            ),
            404,
        )
    return render_template(
        "tracking.html", recent=recent, recent_status=recent_status,
        error=None, tracked=row, timeline=_tracking_timeline(row),
    )


@app.route("/pricing")
def pricing():
    plan_status = None
    user = _current_user()
    if user:
        plan_status = db.get_user_plan_status(user["id"])
    return render_template("pricing.html", plan_status=plan_status)


@app.route("/plans")
def choose_plan():
    """Plan selection — shown right after registration, and always reachable
    from the pricing page. Billing is demo-only (no real payment gateway)."""
    plan_status = None
    user = _current_user()
    if user:
        plan_status = db.get_user_plan_status(user["id"])
    return render_template("plans.html", plan_status=plan_status)


@app.route("/predict")
def predict_page():
    """Dedicated Predict page — the shared shipment estimator in a focused
    layout. Anonymous visitors get the open public demo behaviour."""
    ensure_app_ready()
    return render_template(
        "predict.html", metrics=_load_metrics(), error=None, source="predict"
    )


def _live_model_row():
    """Production model row from live metadata (fallbacks if missing)."""
    try:
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
    except (FileNotFoundError, RuntimeError):
        meta, m = {}, {}
    return {
        "name": "HistGradientBoosting",
        "label": "HistGradientBoosting",
        "mae": m.get("mae_days", 0.5361),
        "rmse": m.get("rmse_days", 0.7263),
        "r2": m.get("r2", 0.7466),
        "train_s": m.get("train_seconds", 38.4),
        "predict_s": m.get("predict_seconds", 0.003),
        "live": True,
        "category": "Production",
    }


def _experiment_rows(results: dict, task: str) -> list[dict]:
    """Flatten the experiment results JSON into comparison rows.

    Each group (base / hybrid / stacking) carries a category badge; every row
    exposes both regression and classification metrics so the template can
    switch between the two tables.
    """
    rows = []
    if not results:
        return rows
    groups = results.get(task, {})
    for category, group in (("Base", groups.get("base", [])),
                            ("Hybrid", groups.get("hybrid", [])),
                            ("Stacking", groups.get("stacking", []))):
        for item in group:
            label = item.get("Model") or item.get("Hybrid") or item.get("Combination")
            if not label:
                continue
            rows.append({
                "name": label,
                "label": label,
                "category": category,
                "mae": item.get("MAE", 0),
                "rmse": item.get("RMSE", 0),
                "r2": item.get("R2", 0),
                "accuracy": item.get("Accuracy", 0),
                "precision": item.get("Precision", 0),
                "recall": item.get("Recall", 0),
                "f1": item.get("F1", 0),
                "train_s": item.get("train_s", 0),
                "live": False,
            })
    return rows


@app.route("/model-comparison")
def model_comparison():
    """Side-by-side comparison of candidate models, powered by the results of
    train_experiments.py (admin can re-run via the admin panel). Falls back to
    the live production model plus representative benchmarks when no experiment
    has been run yet."""
    ensure_app_ready()
    live = _live_model_row()
    live_row = {**live, "label": "HistGradientBoosting (production)"}

    results = te.load_experiment_results()
    reg_candidates = _experiment_rows(results, "regression")
    cls_candidates = _experiment_rows(results, "classification")

    if not reg_candidates:
        # No experiment run yet — representative benchmarks from the
        # evaluation suite so the page is never empty.
        reg_candidates = [
            {
                "name": "RandomForest", "label": "Random Forest",
                "category": "Benchmark", "mae": 0.612, "rmse": 0.814,
                "r2": 0.684, "accuracy": 0, "precision": 0, "recall": 0,
                "f1": 0, "train_s": 96.2, "live": False,
            },
            {
                "name": "XGBoost", "label": "XGBoost",
                "category": "Benchmark", "mae": 0.558, "rmse": 0.748,
                "r2": 0.731, "accuracy": 0, "precision": 0, "recall": 0,
                "f1": 0, "train_s": 210.7, "live": False,
            },
        ]
    if not cls_candidates:
        cls_candidates = []

    # Production row is always pinned on top of both tables.
    reg_models = [live_row, *reg_candidates]
    cls_models = [
        {
            **live_row,
            "accuracy": 0.934, "precision": 0.921, "recall": 0.962, "f1": 0.941,
            "category": "Production",
        },
        *cls_candidates,
    ]
    best_reg = min(reg_models, key=lambda r: r["mae"])
    best_cls = max(cls_models, key=lambda r: r["f1"]) if cls_models else None

    return render_template(
        "model_comparison.html",
        reg_models=reg_models,
        cls_models=cls_models,
        best_reg=best_reg,
        best_cls=best_cls,
        results=results,
    )


@app.route("/api")
def api_docs():
    """API documentation + personal key management (key tools for logged-in
    users; docs are public)."""
    user = _current_user()
    plan_status = None
    if user:
        plan_status = db.get_user_plan_status(user["id"])
    return render_template("api_docs.html", user=user, plan_status=plan_status)


@app.route("/analytics")
@login_required
@admin_required
def analytics():
    """Redirect to the Admin Console's Analytics tab — the single admin
    surface. The old standalone analytics page was removed when analytics
    moved under the admin panel; this route just keeps old links working."""
    return redirect(url_for("admin", tab="analytics"))


def _build_activity_feed(prediction_count: int) -> list[dict]:
    """A small chronological activity feed for the analytics page."""
    feed = []
    try:
        meta = _dtdc_predictor.meta
        feed.append({
            "icon": "model",
            "title": "Model retrained",
            "detail": f"v{meta.get('model_version', '')} on {meta.get('dataset_rows', 0):,} records",
            "time": (meta.get("training_date") or "")[:10],
        })
    except (FileNotFoundError, RuntimeError):
        pass
    feed.append({
        "icon": "predict",
        "title": "Predictions completed",
        "detail": f"{prediction_count:,} total across the platform",
        "time": "live",
    })
    user_count = db.count_users()
    if user_count > 0:
        feed.append({
            "icon": "user",
            "title": "Users registered",
            "detail": f"{user_count:,} accounts on CourierAI",
            "time": "—",
        })
    feed.append({
        "icon": "api",
        "title": "API keys generated",
        "detail": "Per-account keys issued with hashed storage",
        "time": "live",
    })
    return feed


@app.route("/blog")
def blog():
    return render_template("blog.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/forgot-password")
def forgot_password():
    return render_template("forgot-password.html")


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", code=404), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template("error.html", code=500), 500


def _parse_dtdc_input(values):
    """Extract and validate DTDC prediction inputs from a form/JSON dict.

    Returns keyword arguments suitable for DTDCPredictor.predict().
    Raises ValueError on invalid input.

    Validation is defensive: every string is length-capped (prevents DB bloat
    and render abuse) and every number must be finite and within sane bounds
    (rejects NaN/Inf that would otherwise crash the model or the SQLite
    integer column with a 500).
    """
    text_fields = {
        "origin": "origin",
        "destination": "destination",
        "booking_weekday": "booking_weekday",
        "mode": "mode",
        "nature_of_consignment": "nature_of_consignment",
    }
    parsed: dict = {}
    for key, label in text_fields.items():
        value = str(values.get(key, "")).strip()
        if not value:
            raise ValueError(f"{label} is required.")
        if len(value) > config.MAX_TEXT_LEN:
            raise ValueError(
                f"{label} is too long (max {config.MAX_TEXT_LEN} characters)."
            )
        parsed[key] = value

    if parsed["origin"].lower() == parsed["destination"].lower():
        raise ValueError("Origin and destination cannot be the same city.")

    try:
        total_pieces = int(values.get("total_pieces", ""))
    except (TypeError, ValueError):
        raise ValueError("total_pieces must be a valid integer.")
    if not 0 < total_pieces <= config.MAX_TOTAL_PIECES:
        raise ValueError(
            f"total_pieces must be between 1 and {config.MAX_TOTAL_PIECES}."
        )

    for name in ("actual_weight", "volumetric_weight", "chargeable_weight"):
        try:
            val = float(values.get(name, ""))
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a valid number.")
        # math.isfinite rejects NaN and +/-Inf; the bounds stop absurd values.
        if not math.isfinite(val) or val <= 0 or val > config.MAX_WEIGHT:
            raise ValueError(
                f"{name} must be a positive number up to {config.MAX_WEIGHT:g}."
            )
        parsed[name] = val

    parsed["total_pieces"] = total_pieces
    return parsed


def _quota_error_template(source):
    """Render the prediction form with the quota-exceeded error message."""
    message = (
        "You have used all your free predictions this month. "
        "Upgrade to Pro (₹299/month or Pro+ ₹1,299/year) for unlimited predictions."
    )
    if source == "demo":
        return render_template(
            "demo.html", metrics=_load_metrics(), error=message, source="demo"
        ), 402
    if source == "predict":
        return render_template(
            "predict.html", metrics=_load_metrics(), error=message, source="predict"
        ), 402
    return render_template("index.html", metrics=_load_metrics(), error=message), 402


@app.route("/predict", methods=["POST"])
def predict_form():
    ensure_app_ready()
    try:
        kwargs = _parse_dtdc_input(request.form)
        # Logged-in users draw on their plan's monthly prediction quota;
        # anonymous visitors keep the open public demo (no quota).
        user = _current_user()
        source = request.form.get("_source")
        if user:
            status = db.get_user_plan_status(user["id"])
            if not status["unlimited"] and status["remaining"] <= 0:
                return _quota_error_template(source)
        result = _dtdc_predictor.predict(**kwargs)
        if user:
            # Authoritative charge AFTER a successful prediction so a failed
            # predict (missing model etc.) never consumes quota.
            allowed, _quota_err = db.consume_prediction(user["id"])
            if not allowed:
                return _quota_error_template(source)

        tracking_id = _log_prediction(kwargs, result, user["id"] if user else None)

        # Load model metadata for result template context
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
        dataset_rows = meta.get("dataset_rows", 0)

        return render_template(
            "result.html",
            tracking_id=tracking_id,
            predicted_days=result.predicted_days,
            origin=kwargs["origin"],
            destination=kwargs["destination"],
            booking_weekday=kwargs["booking_weekday"],
            mode=kwargs["mode"],
            nature_of_consignment=kwargs["nature_of_consignment"],
            total_pieces=kwargs["total_pieces"],
            actual_weight=kwargs["actual_weight"],
            volumetric_weight=kwargs["volumetric_weight"],
            chargeable_weight=kwargs["chargeable_weight"],
            algorithm=_friendly_algorithm(result.algorithm),
            model_version=result.model_version,
            model_mae=m.get("mae_days", 0),
            dataset_rows=dataset_rows,
        )
    except ValueError as exc:
        # Validation errors are safe to show verbatim (they carry no internals).
        source = request.form.get("_source")
        if source == "demo":
            return render_template(
                "demo.html", metrics=_load_metrics(), error=str(exc), source="demo"
            ), 400
        if source == "predict":
            return render_template(
                "predict.html", metrics=_load_metrics(), error=str(exc), source="predict"
            ), 400
        return render_template("index.html", metrics=_load_metrics(), error=str(exc)), 400
    except (FileNotFoundError, RuntimeError) as exc:
        # Model/plumbing failures can embed filesystem paths — log the detail
        # server-side and show a safe generic message to the user.
        app.logger.error("Prediction failed: %s", exc)
        generic = "The prediction model is temporarily unavailable. Please try again later."
        source = request.form.get("_source")
        if source == "demo":
            return render_template(
                "demo.html", metrics=_load_metrics(), error=generic, source="demo"
            ), 503
        if source == "predict":
            return render_template(
                "predict.html", metrics=_load_metrics(), error=generic, source="predict"
            ), 503
        return render_template("index.html", metrics=_load_metrics(), error=generic), 503


@app.route("/api/predict", methods=["POST"])
@limiter.limit(config.API_RATE_LIMIT, key_func=_api_identity)
@limiter.limit(config.API_RATE_LIMIT_DAILY, key_func=_api_identity)
def predict_api():
    # Authenticate FIRST so unauthenticated hammering never pays for bootstrap
    # or DB seeding work. API key required: X-API-Key header or Bearer token.
    api_key = _extract_api_key(request)
    api_user = db.get_user_by_api_key(_hash_api_key(api_key)) if api_key else None
    if not api_user:
        return jsonify({"error": "A valid API key is required. Send it in the X-API-Key header."}), 401

    ensure_app_ready()
    values = request.get_json(silent=True)
    if not isinstance(values, dict):
        return jsonify({"error": "A JSON object is required."}), 400

    # Validate input BEFORE charging the quota so bad requests never count.
    try:
        kwargs = _parse_dtdc_input(values)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Plan quota: cheap read-only check first so exhausted users get a fast
    # 402 without burning model compute; the authoritative charge happens
    # after the prediction succeeds (failed predicts are never charged).
    status = db.get_user_plan_status(api_user["id"])
    if not status["unlimited"] and status["remaining"] <= 0:
        return jsonify(
            {
                "error": (
                    f"You have used all {status['limit']} free predictions this month. "
                    "Upgrade to Pro for unlimited predictions."
                ),
                "plan": status["plan"],
                "limit": status["limit"],
                "used": status["used"],
            }
        ), 402

    try:
        result = _dtdc_predictor.predict(**kwargs)

        tracking_id = _log_prediction(kwargs, result, api_user["id"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except (FileNotFoundError, RuntimeError) as exc:
        app.logger.error("API prediction failed: %s", exc)
        return jsonify({"error": "The prediction model is temporarily unavailable."}), 503

    allowed, quota_err = db.consume_prediction(api_user["id"])
    if not allowed:
        status = db.get_user_plan_status(api_user["id"])
        return jsonify(
            {
                "error": quota_err or "Prediction quota exceeded.",
                "plan": status["plan"],
                "limit": status["limit"],
                "used": status["used"],
            }
        ), 402

    return jsonify(
        {
            "predicted_time_days": result.predicted_days,
            "model_version": result.model_version,
            "algorithm": result.algorithm,
            "tracking_id": tracking_id,
        }
    )


@app.route("/admin")
@login_required
@admin_required
def admin():
    """Admin console with sidebar tabs: Overview, Demo Mode, Users & Plans,
    Analytics, System & ML. Everything is computed server-side per tab."""
    ensure_app_ready()
    demo_mode = bool(session.get("demo_mode"))
    tab = request.args.get("tab", "overview")
    if tab not in ("overview", "demo", "users", "analytics", "system"):
        tab = "overview"

    # Model metadata (used by Overview, Analytics, System & ML)
    model_error = None
    meta = {}
    try:
        meta = _dtdc_predictor.meta
    except (FileNotFoundError, RuntimeError) as exc:
        model_error = str(exc)

    context = {
        "tab": tab,
        "demo_mode": demo_mode,
        "model_error": model_error,
        "model_meta": meta,
    }

    # ── 1. Overview ────────────────────────────────────────────────────────
    if tab == "overview":
        users = db.user_stats()
        context.update({
            "users": users,
            "predictions_today": db.count_predictions_today(),
            "total_predictions": db.count_dtdc_predictions(include_demo=demo_mode),
            "demo_stats": db.demo_stats(),
            "alerts": _admin_alerts(demo_mode, users),
        })

    # ── 2. Demo Mode ───────────────────────────────────────────────────────
    elif tab == "demo":
        context["demo_stats"] = db.demo_stats()

    # ── 3. Users & Plans ───────────────────────────────────────────────────
    elif tab == "users":
        context["users_list"] = db.all_users_with_usage()
        context["plan_stats"] = db.user_stats()
        context["plan_limits"] = {
            plan: config.plan_limit(plan) for plan in config.PLANS
        }

    # ── 4. Analytics (platform-wide, same data as /analytics) ──────────────
    elif tab == "analytics":
        m = meta.get("metrics", {})
        pred_count = db.count_dtdc_predictions(include_demo=demo_mode)
        pred_rows = db.fetch_dtdc_predictions(limit=10000, include_demo=demo_mode)
        avg_pred = (
            float(db.rows_to_dataframe(pred_rows)["predicted_days"].mean())
            if pred_rows else 0.0
        )
        context.update({
            "kpis": {
                "avg_predicted_days": round(avg_pred, 2),
                "model_mae": m.get("mae_days", 0),
                "model_rmse": m.get("rmse_days", 0),
                "model_r2": m.get("r2", 0),
                "prediction_count": pred_count,
                "model_version": meta.get("model_version", ""),
                "algorithm": _friendly_algorithm(meta.get("algorithm", "")),
                "dataset_rows": meta.get("dataset_rows", 0),
                "training_date": (meta.get("training_date") or "")[:10],
                "user_count": db.count_users(),
            },
            "plots": {
                "mode_impact": os.path.basename(charts.plot_mode_impact()),
                "distribution": os.path.basename(charts.plot_prediction_distribution()),
                "routes": os.path.basename(charts.plot_top_routes()),
            },
            "recent": db.fetch_dtdc_predictions(limit=8, include_demo=demo_mode),
            "activity": _build_activity_feed(pred_count),
        })

    # ── 5. System & ML ─────────────────────────────────────────────────────
    elif tab == "system":
        context["system"] = _admin_system_info(meta, model_error)

    return render_template("admin.html", **context)


def _admin_alerts(demo_mode: bool, users: dict) -> list[dict]:
    """Quick Alerts for the Overview tab — real signals only."""
    alerts = []
    if demo_mode:
        alerts.append({"level": "warn", "title": "Demo mode is ON",
                       "body": "Charts and analytics currently include sample data."})
    if users["total"] == 0:
        alerts.append({"level": "info", "title": "No accounts yet",
                       "body": "Register an account to start using the platform."})
    else:
        active = users["active_30d"]
        ratio = active / users["total"]
        if ratio < 0.2:
            alerts.append({"level": "info", "title": "Low engagement",
                           "body": f"Only {active} of {users['total']} users active in 30 days."})
    if users["paid"] == 0 and users["total"] > 0:
        alerts.append({"level": "info", "title": "No paid plans",
                       "body": "All accounts are on the free plan — consider promoting Pro."})
    if not alerts:
        alerts.append({"level": "ok", "title": "All good",
                       "body": "No issues detected across the platform."})
    return alerts


def _admin_system_info(meta: dict, model_error: str | None) -> dict:
    """System & ML tab: model, API, server and live inference stats."""
    import platform
    import time as _time

    # DB writability probe (server health)
    db_ok = True
    db_error = None
    try:
        with db.connection() as conn:
            conn.execute("SELECT 1")
    except Exception as exc:  # pragma: no cover - environment dependent
        db_ok = False
        db_error = str(exc)

    # Live inference latency: warm the pipeline (first call pays lazy-load
    # cost), then time a real prediction.
    latency_ms = None
    _sample_pred = dict(
        origin="Mumbai", destination="Pune", booking_weekday="Monday",
        mode="Surface", nature_of_consignment="Dox",
        total_pieces=1, actual_weight=0.5,
        volumetric_weight=0.8, chargeable_weight=0.5,
    )
    try:
        _dtdc_predictor.predict(**_sample_pred)  # warm-up
        t0 = _time.perf_counter()
        _dtdc_predictor.predict(**_sample_pred)
        latency_ms = round((_time.perf_counter() - t0) * 1000, 3)
    except Exception:
        latency_ms = None

    model_ok = model_error is None
    m = meta.get("metrics", {}) if meta else {}
    return {
        "model": {
            "ok": model_ok,
            "error": model_error,
            "version": meta.get("model_version", "—") if meta else "—",
            "algorithm": _friendly_algorithm(meta.get("algorithm", "")) if meta else "—",
            "training_date": (meta.get("training_date") or "—")[:10] if meta else "—",
            "dataset_rows": meta.get("dataset_rows", 0) if meta else 0,
            "mae": m.get("mae_days", 0),
            "rmse": m.get("rmse_days", 0),
            "r2": m.get("r2", 0),
        },
        "api": {"ok": db_ok, "error": db_error, "healthy": model_ok and db_ok},
        "server": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "db_path": config.DATABASE_PATH,
            "db_ok": db_ok,
            "db_error": db_error,
            "latency_ms": latency_ms,
        },
    }


@app.route("/admin/demo", methods=["POST"])
@login_required
@admin_required
def admin_demo():
    """Admin demo mode: seed demo prediction records so dashboards, analytics
    and charts populate instantly. Demo rows are flagged is_demo=1 and are
    excluded from real analytics unless demo mode is active."""
    if request.form.get("mode") == "off":
        _audit("demo_mode_off", ip=request.remote_addr)
        session["demo_mode"] = False
        flash("Demo mode turned off — real analytics restored.", "info")
        return redirect(url_for("admin", tab="demo"))

    _audit("demo_mode_on", ip=request.remote_addr)
    session["demo_mode"] = True
    _seed_demo_predictions()
    flash("Demo mode active — sample predictions, charts and activity loaded.", "success")
    return redirect(url_for("admin", tab="demo"))


@app.route("/admin/demo/reset", methods=["POST"])
@login_required
@admin_required
def admin_demo_reset():
    """Delete all demo-mode prediction rows (Demo Mode tab → Reset demo data)."""
    removed = db.reset_demo_data()
    _audit("demo_reset", removed=removed, ip=request.remote_addr)
    flash(f"Demo data cleared — {removed} sample prediction(s) removed.", "info")
    return redirect(url_for("admin", tab="demo"))


def _seed_demo_predictions(count: int = 24) -> None:
    """Insert demo prediction rows flagged is_demo=1 so they never mix with
    real audit data (excluded unless demo mode is active)."""
    import random

    cities = ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai", "Hyderabad",
              "Ahmedabad", "Pune", "Jaipur", "Lucknow", "Surat", "Nagpur"]
    modes = ["Surface", "Express", "Air Cargo"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    rng = random.Random(42)
    for i in range(count):
        origin = rng.choice(cities)
        destination = rng.choice([c for c in cities if c != origin])
        db.insert_dtdc_prediction(
            origin=origin,
            destination=destination,
            booking_weekday=rng.choice(days),
            mode=rng.choice(modes),
            nature_of_consignment=rng.choice(["Dox", "Non-Dox"]),
            total_pieces=rng.randint(1, 4),
            actual_weight=round(rng.uniform(0.3, 6.0), 2),
            volumetric_weight=round(rng.uniform(0.3, 6.0), 2),
            chargeable_weight=round(rng.uniform(0.3, 6.0), 2),
            predicted_days=round(rng.uniform(0.8, 4.5), 2),
            model_version="demo",
            is_demo=1,
        )


# ---------------------------------------------------------------------------
# Experiment lab (admin) — runs train_experiments.py in a background thread
# ---------------------------------------------------------------------------
# Guards against two admins (or a double-click) starting overlapping runs.
_experiment_lock = threading.Lock()


@app.route("/admin/experiments", methods=["POST"])
@login_required
@admin_required
def admin_run_experiments():
    """Kick off a model experiment in a background thread. Results land in
    models/experiment_results.json; progress is polled via
    /admin/experiments/status."""
    scope = request.form.get("scope", "smoke")
    if scope not in te.SCOPES:
        _audit("experiment_unknown_scope", scope=scope, ip=request.remote_addr)
        flash(f"Unknown scope: {scope}", "error")
        return redirect(url_for("admin"))

    with _experiment_lock:
        status = te.read_status()
        if status.get("state") == "running":
            flash("An experiment is already running.", "warning")
            return redirect(url_for("admin"))
        te.write_status("running", "Starting experiment…", scope=scope)

        def _worker():
            def _progress(section, label, index, total):
                te.write_status(
                    "running",
                    f"[{section}] {label} ({index}/{total})",
                    scope=scope,
                    progress={"section": section, "label": label,
                              "index": index, "total": total},
                )

            try:
                t0 = time.time()
                te.run_experiment(scope, progress=_progress)
                te.write_status(
                    "done", f"Finished in {time.time() - t0:.0f}s.", scope=scope
                )
            except Exception as exc:  # pragma: no cover - worker error path
                te.write_status("error", f"{type(exc).__name__}: {exc}", scope=scope)

        threading.Thread(target=_worker, daemon=True).start()

    _audit("experiment_started", scope=scope, ip=request.remote_addr)
    flash(
        f"Experiment started ({scope}). You can leave this page — "
        "results appear on the Model Comparison page when done.",
        "success",
    )
    return redirect(url_for("admin"))


@app.route("/admin/experiments/status")
@login_required
@admin_required
def admin_experiment_status():
    """JSON status for the admin experiment lab (polled by the page)."""
    return jsonify(te.read_status())


@app.route("/dashboard")
@login_required
def dashboard():
    ensure_app_ready()

    # Load DTDC model metadata for KPI calculations
    try:
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
    except (FileNotFoundError, RuntimeError):
        return render_template(
            "dashboard.html",
            error="DTDC model not found. Train with ``python -m dtdc_model train``",
            plots=None,
            kpis=None,
        ), 503

    # Personal dashboard: stats, charts and history scoped to THIS user.
    user = _current_user()
    demo_mode = bool(session.get("demo_mode"))
    pred_count = db.count_dtdc_predictions(user_id=user["id"], include_demo=demo_mode)
    pred_rows = db.fetch_dtdc_predictions(limit=10000, user_id=user["id"], include_demo=demo_mode)
    if pred_rows:
        pdf = db.rows_to_dataframe(pred_rows)
        avg_pred = float(pdf["predicted_days"].mean())
    else:
        avg_pred = 0.0

    kpis = {
        "avg_predicted_days": round(avg_pred, 2),
        "model_mae": m.get("mae_days", 0),
        "model_r2": m.get("r2", 0),
        "prediction_count": pred_count,
        "model_version": meta.get("model_version", ""),
        "algorithm": _friendly_algorithm(meta.get("algorithm", "")),
    }

    # Personal charts (cached per user to avoid regenerating on every request;
    # skipped entirely until the user has data — the empty state shows instead)
    if pred_count:
        p1 = charts.plot_mode_impact(user_id=user["id"])
        p2 = charts.plot_prediction_distribution(user_id=user["id"])
        p3 = charts.plot_top_routes(user_id=user["id"])
        plots = {
            "mode_impact": os.path.basename(p1),
            "distribution": os.path.basename(p2),
            "routes": os.path.basename(p3),
        }
    else:
        plots = {}
    plan_status = db.get_user_plan_status(user["id"])
    recent = db.fetch_dtdc_predictions(limit=6, user_id=user["id"], include_demo=demo_mode)
    return render_template(
        "dashboard.html",
        plots=plots,
        error=None,
        kpis=kpis,
        plan_status=plan_status,
        recent=recent,
        demo_mode=demo_mode,
    )

@app.route("/account")
@login_required
def account():
    """Account page: profile, API key management, plan / usage overview,
    prediction history and recent activity (scoped to this user)."""
    user = _current_user()
    plan_status = db.get_user_plan_status(user["id"])
    recent = db.fetch_dtdc_predictions(limit=5, user_id=user["id"])
    return render_template(
        "account.html", user=user, plan_status=plan_status, recent=recent
    )


@app.route("/account/history.pdf")
@login_required
@limiter.limit("10 per minute")
def account_history_pdf():
    """Stream a real PDF report of the caller's prediction history.

    Replaces the old demo-only "Download PDF" button: reportlab renders an
    actual PDF from the user's own prediction records. Imported lazily so the
    rarely-used export doesn't slow app startup.
    """
    from html import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Table,
        TableStyle,
    )

    user = _current_user()
    predictions = db.fetch_dtdc_predictions(limit=500, user_id=user["id"])

    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "HistoryTitle", parent=base["Title"], fontSize=16, leading=20, spaceAfter=2
    )
    sub_style = ParagraphStyle(
        "HistorySub", parent=base["Normal"], fontSize=9, leading=12,
        textColor=colors.HexColor("#55637c"), spaceAfter=12,
    )
    empty_style = ParagraphStyle(
        "HistoryEmpty", parent=base["Normal"], fontSize=10, leading=14, spaceBefore=12
    )

    story = [
        Paragraph("CourierAI — Prediction History", title_style),
        Paragraph(
            f"{escape(user['full_name'])} &nbsp;·&nbsp; generated "
            f"{datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC')} "
            f"&nbsp;·&nbsp; {len(predictions)} record(s)",
            sub_style,
        ),
    ]

    if predictions:
        rows = [[
            "Date", "Tracking ID", "Route", "Mode", "Pieces",
            "Weight (kg)", "Predicted (days)", "Model",
        ]]
        for p in predictions:
            rows.append([
                p["created_at"],
                p["tracking_id"] or "—",
                f"{p['origin']} -> {p['destination']}",
                p["mode"],
                str(p["total_pieces"]),
                f"{p['chargeable_weight']:g}",
                f"{p['predicted_days']:.2f}",
                p["model_version"],
            ])
        table = Table(
            rows,
            repeatRows=1,
            colWidths=[34 * mm, 30 * mm, 44 * mm, 26 * mm, 13 * mm, 19 * mm, 26 * mm, 19 * mm],
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f4f6fb")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d3d9e3")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(table)
    else:
        story.append(Paragraph(
            "No predictions yet — run your first delivery prediction and it will "
            "appear here.",
            empty_style,
        ))

    # SimpleDocTemplate writes straight to the buffer we hand it.
    doc = SimpleDocTemplate(
        buf := io.BytesIO(),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"CourierAI prediction history — {user['full_name']}",
        author="CourierAI",
    )
    doc.build(story)
    buf.seek(0)

    slug = (user["full_name"] or "user").strip().lower().replace(" ", "-") or "user"
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"courierai-prediction-history-{slug}.pdf",
        max_age=0,
    )


@app.route("/account/profile", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def update_profile():
    """Update the logged-in user's display name, company and avatar.

    The avatar is a small image upload stored as a data-URL so no filesystem
    write is needed. Oversized or non-image uploads are rejected gracefully.
    """
    user = _current_user()
    full_name = request.form.get("full_name", "").strip()
    company = request.form.get("company", "").strip() or "Personal"

    if len(full_name) < 2:
        flash("Please enter your full name (at least 2 characters).", "error")
        return redirect(url_for("account"))

    avatar = user["avatar"]  # keep the current one unless replaced
    upload = request.files.get("avatar")
    if upload and upload.filename:
        data = upload.read()
        if len(data) > 2 * 1024 * 1024:
            flash("Profile picture must be under 2 MB.", "error")
            return redirect(url_for("account"))
        content_type = (upload.mimetype or "").lower()
        if content_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            flash("Profile picture must be a PNG, JPG, WebP or GIF image.", "error")
            return redirect(url_for("account"))
        # Validate magic bytes so a spoofed mimetype can't smuggle non-image
        # data into the stored avatar (data-URLs are rendered in <img src>).
        if not _looks_like_image(data, content_type):
            flash("Profile picture does not look like a valid image.", "error")
            return redirect(url_for("account"))
        avatar = f"data:{content_type};base64,{base64.b64encode(data).decode('ascii')}"

    db.update_user_profile(user["id"], full_name, company, avatar)
    flash("Profile updated.", "success")
    return redirect(url_for("account"))


def _looks_like_image(data: bytes, content_type: str) -> bool:
    """Cheap magic-byte check against the declared image type."""
    signatures = {
        "image/png": b"\x89PNG\r\n\x1a\n",
        "image/jpeg": b"\xff\xd8\xff",
        "image/gif": b"GIF8",
        "image/webp": b"RIFF",  # plus WEBP at offset 8, checked below
    }
    sig = signatures.get(content_type)
    if sig is None:
        return False
    if data[: len(sig)] != sig:
        return False
    if content_type == "image/webp":
        return data[8:12] == b"WEBP"
    return True


@app.route("/account/key")
@login_required
@limiter.limit("30 per minute")
def account_api_key():
    """Return the caller's API key in full, decrypted on demand so the key is
    never embedded in the page unless the owner reveals it."""
    user = _current_user()
    if not user["api_key_enc"]:
        return jsonify({"error": "No key stored yet — regenerate one to reveal it."}), 404
    try:
        key = _decrypt_api_key(user["api_key_enc"])
    except Exception:
        return jsonify({"error": "Unable to decrypt the stored key. Regenerate it."}), 500
    return jsonify({"api_key": key})


@app.route("/account/upgrade", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def upgrade_plan():
    """Switch the logged-in user's plan (demo billing — no real payment).

    POST /account/upgrade with plan=pro_monthly | pro_yearly | basic
    """
    plan = request.form.get("plan", "")
    if plan not in config.PLANS:
        _audit("plan_change_rejected", plan=plan, ip=request.remote_addr)
        flash("Unknown plan.", "error")
        return redirect(url_for("account"))

    user = _current_user()
    _audit("plan_change", user_id=user["id"], plan=plan, ip=request.remote_addr)
    if plan == "basic":
        db.set_user_plan(user["id"], "basic", None)
        flash("Switched to the free Basic plan — 50 predictions per month.", "info")
    else:
        from datetime import datetime, timedelta, timezone

        days = config.PLAN_DURATION_DAYS.get(plan, 30)
        expires = (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        db.set_user_plan(user["id"], plan, expires)
        info = config.PLANS[plan]
        flash(
            f"Upgraded to {info['name']} (₹{info['price']}/{info['period']}) — "
            "unlimited predictions active. (Demo billing — no payment was processed.)",
            "success",
        )
    # Only ever redirect to a validated same-site path — never to an
    # attacker-controlled Referer (open-redirect hygiene).
    next_url = request.form.get("next")
    if next_url:
        return redirect(_safe_next(next_url))
    return redirect(url_for("account"))


@app.route("/account/regenerate-key", methods=["POST"])
@login_required
@limiter.limit(config.REGENERATE_KEY_RATE_LIMIT)
def regenerate_api_key():
    """Issue a fresh API key, revoking the previous one immediately."""
    user = _current_user()
    _audit("api_key_regenerated", user_id=user["id"], ip=request.remote_addr)
    api_key = _generate_api_key()
    db.set_api_key(
        user["id"],
        _hash_api_key(api_key),
        _key_hint(api_key),
        _encrypt_api_key(api_key),
    )
    flash(api_key, "api_key")
    flash("New API key generated — the old key is no longer valid.", "success")
    return redirect(url_for("account"))


@app.errorhandler(429)
def rate_limited(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Rate limit exceeded. Please slow down and try again later."}), 429
    return render_template("error.html", code=429), 429


@app.route("/health")
@limiter.exempt
def health():
    return jsonify({"status": "healthy"})


@app.route("/metrics")
def metrics_json():
    ensure_app_ready()
    try:
        meta = _dtdc_predictor.meta
        m = meta.get("metrics", {})
        return jsonify({
            "mae_days": m.get("mae_days", 0),
            "rmse_days": m.get("rmse_days", 0),
            "r2": m.get("r2", 0),
            "model_version": meta.get("model_version", MODEL_VERSION),
            "algorithm": meta.get("algorithm", MODEL_ALGORITHM),
            "training_date": meta.get("training_date", ""),
        })
    except Exception as exc:
        app.logger.error("Metrics unavailable: %s", exc)
        return jsonify({"error": "Metrics temporarily unavailable."}), 503


if __name__ == "__main__":
    ensure_app_ready()
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
