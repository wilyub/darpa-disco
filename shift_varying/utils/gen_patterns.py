import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_checkerboard(size=256, num_squares=8):
    """Generate a checkerboard pattern."""
    square_size = size // num_squares
    checkerboard = np.zeros((size, size), dtype=np.uint8)
    
    for i in range(num_squares):
        for j in range(num_squares):
            if (i + j) % 2 == 0:
                checkerboard[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = 255
                
    return checkerboard

def generate_grid(size=256, line_spacing=16):
    """Generate a simple grid pattern."""
    grid = np.ones((size, size), dtype=np.uint8) * 255
    for i in range(0, size, line_spacing):
        grid[:, i] = 0  # Vertical lines
        grid[i, :] = 0  # Horizontal lines
    return grid

def generate_starburst(size=256, num_lines=32):
    """Generate a radial starburst pattern."""
    starburst = np.ones((size, size), dtype=np.uint8) * 255
    center = (size // 2, size // 2)
    
    for i in range(num_lines):
        angle = (i / num_lines) * np.pi
        x = int(center[0] + np.cos(angle) * size)
        y = int(center[1] + np.sin(angle) * size)
        cv2.line(starburst, center, (x, y), 0, 1)
    
    return starburst

def generate_dot_grid(size=256, spacing=32, dot_radius=3):
    """Generate a uniform dot grid pattern."""
    dot_grid = np.ones((size, size), dtype=np.uint8) * 255
    for i in range(0, size, spacing):
        for j in range(0, size, spacing):
            cv2.circle(dot_grid, (j, i), dot_radius, 0, -1)
    return dot_grid

def show_images():
    """Display generated images."""
    images = {
        "Checkerboard": generate_checkerboard(),
        "Grid": generate_grid(),
        "Starburst": generate_starburst(),
        "Dot Grid": generate_dot_grid()
    }

    plt.figure(figsize=(10, 10))
    for i, (title, img) in enumerate(images.items()):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Show the images
    show_images()
