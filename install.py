#!/usr/bin/env python3
"""
Installation script for SimPEG DC PyTorch modifications.

This script overwrites the standard SimPEG installation with PyTorch-enabled versions
of the core modules, specifically optimized for DC resistivity simulations.

Prerequisites:
    - Active conda environment with SimPEG 0.18.1 and PyTorch 2.5.1 installed
    - pydiso installed via conda-forge
"""

import shutil
import sys
from pathlib import Path
import importlib.util


def find_simpeg_installation():
    """Find the location of the installed SimPEG package."""
    try:
        import SimPEG
        simpeg_path = Path(SimPEG.__file__).parent
        return simpeg_path
    except ImportError:
        print("ERROR: SimPEG not found. Please ensure conda environment is activated and SimPEG is installed:")
        print("  conda install -c conda-forge simpeg=0.18.1")
        sys.exit(1)


def find_discretize_installation():
    """Find the location of the installed discretize package."""
    try:
        import discretize
        discretize_path = Path(discretize.__file__).parent
        return discretize_path
    except ImportError:
        print("ERROR: discretize not found. This should be installed with SimPEG.")
        sys.exit(1)


def get_modifications_path():
    """Get the path to the PyTorch modifications."""
    current_file = Path(__file__)
    mods_path = current_file.parent / "DC_torch"
    
    if not mods_path.exists():
        print(f"ERROR: Modifications not found at {mods_path}")
        print("Please ensure DC_torch/ directory exists in the repository root.")
        sys.exit(1)
    
    return mods_path


def selective_replace_discretize(torch_discretize_path, vanilla_discretize_path):
    """
    Selectively replace discretize files/folders that exist in torch version.
    Preserves all vanilla files/folders that don't exist in torch version.
    This is crucial for preserving compiled extensions.
    """
    print(f"Selective discretize update from: {torch_discretize_path}")
    
    # Process all items in torch discretize
    for torch_item in torch_discretize_path.iterdir():
        # Skip __pycache__ directories
        if torch_item.name == "__pycache__":
            continue
            
        vanilla_item = vanilla_discretize_path / torch_item.name
        
        # Only replace if the item exists in vanilla
        if vanilla_item.exists():
            print(f"   Replacing: {torch_item.name}")
            
            if torch_item.is_file():
                # Replace file
                shutil.copy2(torch_item, vanilla_item)
            elif torch_item.is_dir():
                # Replace entire directory
                shutil.rmtree(vanilla_item)
                shutil.copytree(torch_item, vanilla_item)
        else:
            print(f"   Skipping: {torch_item.name} (not in vanilla)")


def install_pytorch_modifications():
    """Install PyTorch modifications by overwriting SimPEG files."""
    print("SimPEG DC PyTorch Installation")
    print("=" * 50)
    
    # Find installation paths
    print("Step 1: Locating packages...")
    simpeg_path = find_simpeg_installation()
    discretize_path = find_discretize_installation()
    mods_path = get_modifications_path()
    
    print(f"SimPEG installation found at: {simpeg_path}")
    print(f"Discretize installation found at: {discretize_path}")
    print(f"PyTorch modifications found at: {mods_path}")
    
    # Install modifications
    print("\nStep 2: Installing PyTorch modifications...")
    
    # Complete SimPEG replacement
    simpeg_mods = mods_path / "SimPEG"
    if simpeg_mods.exists():
        print("Complete SimPEG replacement")
        shutil.copytree(simpeg_mods, simpeg_path, dirs_exist_ok=True)
    else:
        print("WARNING: SimPEG modifications not found")
    
    # Selective discretize replacement (preserves compiled extensions)
    discretize_mods = mods_path / "discretize"
    if discretize_mods.exists():
        selective_replace_discretize(discretize_mods, discretize_path)
    else:
        print("WARNING: discretize modifications not found")
    
    # Install solver and utils as independent packages in site-packages
    site_packages = simpeg_path.parent  # Get site-packages directory
    
    solver_mods = mods_path / "solver"
    if solver_mods.exists():
        solver_target = site_packages / "solver"
        print("Installing solver as independent package")
        shutil.copytree(solver_mods, solver_target, dirs_exist_ok=True)
    else:
        print("INFO: solver modifications not found (optional)")
    
    utils_mods = mods_path / "utils"
    if utils_mods.exists():
        utils_target = site_packages / "utils"
        print("Installing utils as independent package")
        shutil.copytree(utils_mods, utils_target, dirs_exist_ok=True)
    else:
        print("INFO: utils modifications not found (optional)")
    
    print("\nInstallation completed successfully!")
    print("\nNext steps:")
    print("1. Test the installation:")
    print("   python -c 'import SimPEG; print(\"SimPEG PyTorch ready!\")'")
    print("2. Check PyTorch backend:")
    print("   python -c 'from SimPEG.config import SimpegConfig; print(f\"PyTorch active: {SimpegConfig().torch_is_active}\")'")
    print("\nTo restore original packages, reinstall with conda:")
    print("   conda install --force-reinstall simpeg=0.18.1")


def main():
    """Main entry point for the installation script."""
    try:
        install_pytorch_modifications()
    except KeyboardInterrupt:
        print("\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nInstallation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()