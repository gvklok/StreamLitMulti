# Streamlit Apps Dashboard

A simple dashboard that launches 10 standalone Streamlit apps, each living in its own folder.

- Dashboard file: `dashboard.py`
- App folders: `Project1` … `Project10`, each with its own `app.py` and README.

## Quick Start (Local)

- macOS, zsh assumed
- Python 3.9+ recommended

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Run the dashboard

```bash
streamlit run dashboard.py --server.port 8501
```

- Clicking a button starts an app on its own port (8601–8610) and shows its URL.

### 4) Stop apps

- Stop the dashboard with Ctrl+C. If a port remains busy:

```bash
lsof -i :8601
kill -9 <PID>
```

## Deploying to Streamlit Cloud

There are two recommended deployment patterns:

- Single dashboard app that LINKS to other deployed apps (recommended)
- Ten separate apps deployed individually (one per `ProjectX` folder)

The provided `dashboard.py` automatically detects cloud mode via `st.secrets["APP_URLS"]`. When present, it shows link buttons to your deployed apps instead of trying to spawn subprocesses (which is not supported on Streamlit Cloud).

### Configure secrets for the dashboard

In Streamlit Cloud, set Secrets for the dashboard app like this:

```toml
[APP_URLS]
Project1 = "https://<your-user>-<your-repo>-project1.streamlit.app"
Project2 = "https://<your-user>-<your-repo>-project2.streamlit.app"
Project3 = "https://<your-user>-<your-repo>-project3.streamlit.app"
Project4 = "https://<your-user>-<your-repo>-project4.streamlit.app"
Project5 = "https://<your-user>-<your-repo>-project5.streamlit.app"
Project6 = "https://<your-user>-<your-repo>-project6.streamlit.app"
Project7 = "https://<your-user>-<your-repo>-project7.streamlit.app"
Project8 = "https://<your-user>-<your-repo>-project8.streamlit.app"
Project9 = "https://<your-user>-<your-repo>-project9.streamlit.app"
Project10 = "https://<your-user>-<your-repo>-project10.streamlit.app"
```

See `secrets.example.toml` for a ready-to-copy template.

### How to deploy

1. Push this repo to GitHub.
2. In Streamlit Cloud, deploy apps:
   - Dashboard: set entry point to `dashboard.py` at repo root.
   - Each ProjectX app: set working dir to `ProjectX/` and entry point to `app.py`.
3. After deploying each ProjectX app, copy its URL into the dashboard secrets under `[APP_URLS]`.

### Notes

- Do NOT rely on spawning subprocesses in the cloud; use links.
- Keep `requirements.txt` at repo root so all apps pick up dependencies.
- If a project needs extra packages, add them to the root `requirements.txt` (shared by all apps) or split repos per app.

## Structure

```
Streamlit-App-Dashboard/
├── dashboard.py
├── requirements.txt
├── secrets.example.toml
├── Project1/
│   ├── app.py
│   └── README.md
├── Project2/
│   ├── app.py
│   └── README.md
... (up to Project10)
```

Each project is fully standalone; you can also run any one directly:

```bash
cd Project3
streamlit run app.py --server.port 8603
```
