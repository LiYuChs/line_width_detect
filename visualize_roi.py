import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ROI parameters
ROI_START = 350
ROI_END = 550

def visualize_roi(image_path):
    """Visualize the ROI region on the image."""
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    h, w = img.shape
    mid_x = w // 2
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Full image with ROI rectangle
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray')
    # Draw ROI rectangle
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, ROI_START), w, ROI_END - ROI_START, 
                      linewidth=2, edgecolor='red', facecolor='none', label='ROI')
    ax1.add_patch(rect)
    # Draw center line
    ax1.axvline(x=mid_x, color='green', linewidth=1, linestyle='--', label='Center line')
    ax1.set_title('Full Image with ROI Region', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Pixel (X)')
    ax1.set_ylabel('Pixel (Y)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Zoomed ROI region
    ax2 = axes[1]
    roi_region = img[ROI_START:ROI_END, :]
    ax2.imshow(roi_region, cmap='gray', extent=[0, w, ROI_END, ROI_START])
    # Draw center line in zoomed view
    ax2.axvline(x=mid_x, color='green', linewidth=1, linestyle='--', label='Center line')
    ax2.set_title(f'ROI Region (Y: {ROI_START}-{ROI_END})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Pixel (X)')
    ax2.set_ylabel('Pixel (Y)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path.cwd() / 'data' / 'results' / 'roi_visualization.png', dpi=150, bbox_inches='tight')
    print(f"ROI visualization saved to data/results/roi_visualization.png")
    plt.show()
    
    # Print ROI statistics
    print(f"\nROI Region Information:")
    print(f"  ROI Y Range: {ROI_START} - {ROI_END} (Height: {ROI_END - ROI_START} pixels)")
    print(f"  Center X: {mid_x}")
    print(f"  Image Dimensions: {w} x {h}")

if __name__ == "__main__":
    # Use the first image as example
    image_path = Path.cwd() / 'data' / 'measure_01.jpg'
    visualize_roi(str(image_path))
