import numpy as np
from PIL import Image, ImageDraw
import random
import os
import math

def create_leaf_shape(draw, center_x, center_y, size, angle=0):
    """Create a simple leaf shape using ellipses and lines"""
    # Rotate coordinates
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    
    # Main leaf body (ellipse)
    width, height = size, size * 0.6
    x1 = center_x - width//2
    y1 = center_y - height//2
    x2 = center_x + width//2
    y2 = center_y + height//2
    
    # Draw main leaf body
    leaf_color = random.randint(40, 120)  # Gray values for leaf
    draw.ellipse([x1, y1, x2, y2], fill=leaf_color)
    
    # Draw central vein
    vein_color = max(10, leaf_color - 30)
    draw.line([center_x, y1, center_x, y2], fill=vein_color, width=1)
    
    # Draw side veins
    for i in range(3):
        offset_y = (i + 1) * height // 4
        y_pos = y1 + offset_y
        vein_length = width // 3
        draw.line([center_x - vein_length//2, y_pos, center_x + vein_length//2, y_pos], 
                 fill=vein_color, width=1)

def create_background_texture(width, height):
    """Create a random background texture"""
    # Create noise background
    noise = np.random.randint(180, 255, (height, width), dtype=np.uint8)
    
    # Add some random patterns
    for _ in range(random.randint(10, 30)):  # More patterns for larger image
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        radius = random.randint(1, 5)  # Larger radius for 64x64
        intensity = random.randint(160, 220)
        
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if 0 <= x+dx < width and 0 <= y+dy < height:
                    if dx*dx + dy*dy <= radius*radius:
                        noise[y+dy, x+dx] = intensity
    
    return noise

def generate_image_with_leaf():
    """Generate a 64x64 image with a leaf"""
    # Create background
    background = create_background_texture(64, 64)
    img = Image.fromarray(background, 'L')
    draw = ImageDraw.Draw(img)
    
    # Add 1-3 leaves
    num_leaves = random.randint(1, 3)
    for _ in range(num_leaves):
        center_x = random.randint(12, 52)  # Adjusted for 64x64
        center_y = random.randint(12, 52)  # Adjusted for 64x64
        size = random.randint(16, 28)      # Larger size for 64x64
        angle = random.uniform(0, 2 * math.pi)
        
        create_leaf_shape(draw, center_x, center_y, size, angle)
    
    return img

def generate_image_without_leaf():
    """Generate a 64x64 image without a leaf"""
    # Create background
    background = create_background_texture(64, 64)
    img = Image.fromarray(background, 'L')
    draw = ImageDraw.Draw(img)
    
    # Add random geometric shapes that are not leaves
    num_shapes = random.randint(0, 3)
    for _ in range(num_shapes):
        shape_type = random.choice(['circle', 'rectangle', 'line'])
        color = random.randint(50, 150)
        
        if shape_type == 'circle':
            x = random.randint(4, 60)       # Adjusted for 64x64
            y = random.randint(4, 60)       # Adjusted for 64x64
            radius = random.randint(4, 12)  # Larger radius
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
        elif shape_type == 'rectangle':
            x1 = random.randint(4, 48)      # Adjusted for 64x64
            y1 = random.randint(4, 48)      # Adjusted for 64x64
            x2 = x1 + random.randint(6, 16) # Larger rectangles
            y2 = y1 + random.randint(6, 16) # Larger rectangles
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:  # line
            x1 = random.randint(4, 60)      # Adjusted for 64x64
            y1 = random.randint(4, 60)      # Adjusted for 64x64
            x2 = random.randint(4, 60)      # Adjusted for 64x64
            y2 = random.randint(4, 60)      # Adjusted for 64x64
            draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 3))
    
    return img

def generate_dataset(num_images=4000, output_dir="../sym_dataset"):
    """Generate dataset with approximately 50% leaf images"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    leaf_count = 0
    no_leaf_count = 0
    
    for i in range(num_images):
        # 50% chance of generating leaf image
        has_leaf = random.random() < 0.5
        
        if has_leaf:
            img = generate_image_with_leaf()
            filename = f"LEAF_{leaf_count:04d}.png"
            leaf_count += 1
        else:
            img = generate_image_without_leaf()
            filename = f"NO_{no_leaf_count:04d}.png"
            no_leaf_count += 1
        
        # Save image
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_images} images")
    
    print(f"Dataset generation complete!")
    print(f"Leaf images: {leaf_count}")
    print(f"No-leaf images: {no_leaf_count}")
    print(f"Images saved in: {output_dir}")

if __name__ == "__main__":
    generate_dataset(num_images=4000, output_dir="../sym_dataset")
