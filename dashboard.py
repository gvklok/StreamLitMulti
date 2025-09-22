import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
PROJECTS = [f"Project{i}" for i in range(1, 11)]
BASE_PORT = 8601  # avoid clashing with the dashboard's own port

# Cloud mode: if APP_URLS provided in secrets, we link to deployed apps instead of spawning processes
try:
    APP_URLS = dict(st.secrets.get("APP_URLS", {}))
except Exception:
    APP_URLS = {}
CLOUD_MODE = bool(APP_URLS)

if "processes" not in st.session_state:
    st.session_state.processes = {}

st.set_page_config(page_title="Apps Dashboard", page_icon="🗂️", layout="centered")
st.title("Apps Dashboard")

if CLOUD_MODE:
    st.caption("Cloud mode: open deployed apps via links.")
    for name in PROJECTS:
        url = APP_URLS.get(name)
        col1, col2 = st.columns([1, 2])
        with col1:
            if url:
                if hasattr(st, "link_button"):
                    st.link_button(name, url)
                else:
                    st.markdown(f"[{name}]({url})")
            else:
                st.button(name, disabled=True)
        with col2:
            if url:
                st.write(url)
            else:
                st.warning("URL not configured in secrets.")
    st.divider()
    st.caption("Configure [APP_URLS] in Streamlit secrets to enable links.")
else:
    st.caption("Local mode: launch standalone apps on their own ports.")

    python_cmd = sys.executable or "python3"

    for idx, name in enumerate(PROJECTS, start=0):
        port = BASE_PORT + idx
        app_path = ROOT / name / "app.py"

        if not app_path.exists():
            continue

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button(name, key=f"btn-{name}"):
                proc = st.session_state.processes.get(name)
                if proc is None or proc.poll() is not None:
                    cmd = [
                        python_cmd,
                        "-m",
                        "streamlit",
                        "run",
                        str(app_path),
                        "--server.port",
                        str(port),
                        "--server.headless",
                        "true",
                        "--server.address",
                        "localhost",
                    ]
                    env = os.environ.copy()
                    env.setdefault("PYTHONUNBUFFERED", "1")
                    st.session_state.processes[name] = subprocess.Popen(
                        cmd,
                        cwd=str(app_path.parent),
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    st.info(f"Starting {name} on port {port}…")
        with col2:
            url = f"http://localhost:{port}"
            st.markdown(f"[Open {name}]({url})")

    st.divider()
    st.caption("Tip: Each project is a standalone Streamlit app inside its folder.")
