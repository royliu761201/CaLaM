
import subprocess
import sys
import os

def main():
    print("🚀 Triggering Remote RTP Download (Direct)...")

    script_path = os.path.join(os.path.dirname(__file__), "download_rtp_direct.sh")
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        sys.exit(1)

    # Ensure executable
    subprocess.run(["chmod", "+x", script_path])

    print(f"   Executing: bash {script_path}")
    result = subprocess.run(["bash", "-c", f"source {script_path}"], capture_output=True, text=True)

    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)

    if result.returncode != 0:
        print("❌ Download Failed!")
        sys.exit(result.returncode)
    else:
        print("✅ Download Complete!")

if __name__ == "__main__":
    main()
