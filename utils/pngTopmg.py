from PIL import Image
import glob
import os

def convert_png_to_pgm(png_path, pgm_path, size=(256, 256)):
    img = Image.open(png_path).convert('L').resize(size)
    width, height = img.size
    pixels = list(img.getdata())

    with open(pgm_path, 'w') as f:
        f.write("P2\n")
        f.write(f"{width} {height}\n")
        f.write("255\n")
        for i in range(height):
            row = pixels[i * width:(i + 1) * width]
            f.write(" ".join(map(str, row)) + "\n")

# Convert all PNG images to PGM format
def convert_all_images():
    # Get the script directory and construct absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_data_dir = os.path.join(script_dir, "..", "pca", "imgData")
    pgm_output_dir = os.path.join(img_data_dir, "pgm")
    
    # Create output directory for PGM files
    os.makedirs(pgm_output_dir, exist_ok=True)
    
    # Find all PNG files in the pca/imgData directory
    png_pattern = os.path.join(img_data_dir, "face_*.png")
    png_files = glob.glob(png_pattern)
    
    print(f"Searching in: {img_data_dir}")
    print(f"Pattern: {png_pattern}")
    print(f"Found {len(png_files)} PNG files to convert")
    
    # List all files in the directory for debugging
    if len(png_files) == 0:
        print("Debug: Files in imgData directory:")
        try:
            all_files = os.listdir(img_data_dir)
            for file in all_files:
                print(f"  - {file}")
        except FileNotFoundError:
            print(f"Directory {img_data_dir} does not exist!")
    
    for png_path in png_files:
        try:
            # Extract filename without extension
            filename = os.path.basename(png_path).replace('.png', '')
            pgm_path = os.path.join(pgm_output_dir, f"{filename}.pgm")
            
            convert_png_to_pgm(png_path, pgm_path)
            print(f"[+] Converted {filename}.png to {filename}.pgm")
            
        except Exception as e:
            print(f"[!] Failed to convert {png_path}: {e}")

if __name__ == "__main__":
    convert_all_images()
