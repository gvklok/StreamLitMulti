import os
import subprocess
import sys
import socket
import time
from pathlib import Path

import streamlit as st

def check_port_in_use(port):
    """Check if a port is already in use"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return False
    except OSError:
        return True

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
                # Check if port is already in use
                if check_port_in_use(port):
                    st.warning(f"⚠️ Port {port} is already in use. {name} might already be running!")
                    st.info("If the app is already running, just click the link below.")
                else:
                    proc = st.session_state.processes.get(name)
                    if proc is None or proc.poll() is not None:
                        cmd = [
                            python_cmd,
                            "-m",
                            "streamlit",
                            "run",
                            "app.py",  # Use just app.py since we're running from the project directory
                            "--server.port",
                            str(port),
                            "--server.headless",
                            "true",
                            "--server.address",
                            "localhost",
                        ]
                        env = os.environ.copy()
                        env.setdefault("PYTHONUNBUFFERED", "1")
                        try:
                            process = subprocess.Popen(
                                cmd,
                                cwd=str(app_path.parent),
                                env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                            st.session_state.processes[name] = process
                            st.success(f"✅ Successfully started {name} on port {port}")
                            st.info("Wait a few seconds for the app to fully load, then click the link below.")
                            
                            # Check if process started successfully
                            import time
                            time.sleep(1)
                            if process.poll() is not None:
                                stdout, stderr = process.communicate()
                                st.error(f"❌ {name} failed to start:")
                                if stderr:
                                    st.code(stderr, language="text")
                                if stdout:
                                    st.code(stdout, language="text")
                        except Exception as e:
                            st.error(f"❌ Error launching {name}: {str(e)}")
        with col2:
            url = f"http://localhost:{port}"
            st.markdown(f"[Open {name}]({url})")

    st.divider()
    st.caption("Tip: Each project is a standalone Streamlit app inside its folder.")
