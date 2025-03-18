from PIL import Image

# Open an image file
image_path = 'himanshu_suryawanshi.jpg'  # Replace with your JPG image path
image = Image.open(image_path)

# Check the image mode (should be RGB for JPG images)
print(f"Image mode: {image.mode}")

# If the image is not in RGB, convert it to RGB (just in case)
if image.mode != 'RGB':
    image = image.convert('RGB')

# Save the image after ensuring it's in RGB format
image.save('converted_image.jpg', 'JPEG')

# You can also perform any image processing, like resizing or cropping, here
image_resized = image.resize((800, 800))  # Resize to 800x800 pixels
image_resized.save('resized_image.jpg', 'JPEG')

# Display the image
image.show()
