#!/usr/bin/env python
"""
Post-installation script that ensures sam-2 correctly overwrites hq-sam-2 files.
This is important since SAM2 has developed further after SAM2HQ was released.
However SAM2HQ uses the same internal package names and therefore is not anymore compatible
with a concurrent installation of both packages.

I am "solving" this here by force reinstalling sam-2 after hq-sam-2 is installed.
"""

import subprocess
import sys


def ensure_pip():
    """Ensure pip is available, installing it if necessary."""
    try:
        import pip
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip"])
            return True
        except Exception as e:
            print(f"Warning: Could not ensure pip is installed: {e}")
            return False

def run_post_install():
    """Run post-installation tasks."""
    if not ensure_pip():
        print("Cannot complete post-installation: pip is not available")
        return 1
    
    print("Making sure sam-2 overwrites hq-sam-2 files...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--force-reinstall", "git+https://github.com/horsto/sam2.git"
        ])
        print("Successfully installed sam-2 with overwrite.")
        return 0
    except Exception as e:
        print(f"Error installing sam-2: {e}")
        return 1

def main():
    """Main entry point for the post-installation script."""
    return run_post_install()

if __name__ == "__main__":
    sys.exit(main())
