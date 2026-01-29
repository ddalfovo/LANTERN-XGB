"""
Plotting Utilities Module
-------------------------
This module provides standardized functions for saving Matplotlib figures. 
It ensures consistent naming conventions based on project configuration 
and exports plots in both raster (PNG) and vector (SVG) formats.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import yaml

# --- CONFIGURATION LOADING ---
# Load global settings from the project configuration file.
# Expects 'TARGET_COLUMN' to be defined in the YAML.
try:
    with open("./scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Warning: config.yml not found. Ensure the path is correct.")
    config = {"TARGET_COLUMN": "default"}

def save_plot(fig, title, out_dir):
    """
    Standardizes the saving of Matplotlib figures to a specified directory.

    This function automates the creation of the output directory, generates 
    a sanitized filename using the target variable and plot title, and 
    saves the figure in both high-resolution PNG and scalable SVG formats.

    Args:
        fig (matplotlib.figure.Figure): The figure object to be saved.
        title (str): Descriptive title for the plot (used in the filename).
        out_dir (str or Path): The directory where the files will be stored.

    Returns:
        None
    """
    # Convert string path to a Path object for robust filesystem handling
    out_dir = Path(out_dir)
    
    # Create the directory if it doesn't exist (including parent folders)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a sanitized base name: {TARGET_COLUMN}_{title_with_underscores}
    base_name = f"{config['TARGET_COLUMN']}_{title.replace(' ', '_').lower()}"

    # 1. Save as PNG: Best for presentations and quick previews (300 DPI for print quality)
    png_path = out_dir / f"{base_name}.png"
    fig.savefig(png_path, format='png', dpi=300) 
    
    # 2. Save as SVG: Best for high-quality publications and further editing
    svg_path = out_dir / f"{base_name}.svg"
    fig.savefig(svg_path, format='svg')

    # Log the successful export
    print(f"Saved plots: \n - {png_path}\n - {svg_path}")
    
    # Close the figure to free up memory/prevent overlapping plots in loops
    plt.close(fig)