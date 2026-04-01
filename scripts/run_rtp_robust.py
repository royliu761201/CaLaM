
import subprocess
import sys
import os

def main():
    print("🚀 Triggering Remote RTP Download (Robust Python)...")

    script_module = "projects.calam.download_rtp_robust"

    # We want to run this module using python -m
    # The environment should already be set by run_remote_gpu.py (conda activate, PYTHONPATH)
    # We just need to make sure we don't override HF_HUB_OFFLINE in a way that breaks it, 
    # but the script itself sets it to 0.

    # Actually, we are running THIS script locally inside run_remote_gpu.py's logic?
    # NO. run_remote_gpu.py runs THIS script ON THE REMOTE.
    # So we can just import the logic or subprocess it.

    # Simplest way: just import the main function from the robust script if it's in the path
    try:
        from projects.calam.download_rtp_robust import main as run_download
        run_download()
    except ImportError:
        # Fallback to subprocess if import fails (e.g. path issues)
        cmd = [sys.executable, "-m", "projects.calam.download_rtp_robust"]
        subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
