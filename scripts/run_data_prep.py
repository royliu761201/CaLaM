
import subprocess
import sys
import os

def main():
    print("🚀 Triggering Remote Data Preparation...")
    # This script is intended to be run ON THE REMOTE by run_remote_gpu.py

    prep_script = os.path.join(os.path.dirname(__file__), "scripts", "prepare_data.sh")
    if not os.path.exists(prep_script):
        print(f"❌ Prep script not found at {prep_script}")
        sys.exit(1)

    print(f"   Executing: bash {prep_script}")
    # Force bash execution
    result = subprocess.run(["bash", prep_script], capture_output=True, text=True)

    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)

    if result.returncode != 0:
        print("❌ Data Preparation Failed!")
        sys.exit(result.returncode)
    else:
        print("✅ Data Preparation Complete!")

if __name__ == "__main__":
    main()
