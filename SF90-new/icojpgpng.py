from PIL import Image
import os

def convert_to_png(input_path, output_path):
    try:
        with Image.open(input_path) as img:
          
            img.save(output_path, 'PNG')
            print(f"Conversion successful: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error converting {input_path} to PNG: {e}")

def batch_convert_to_png(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

     
        if os.path.isdir(input_path):
            continue


        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")

       
        convert_to_png(input_path, output_path)

# Convert to Png 
input_folder = r"Test-Data/in ico"
output_folder = r"Test-Data/out png"
batch_convert_to_png(input_folder, output_folder)

