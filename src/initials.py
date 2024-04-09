from scipy.interpolate import griddata
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.interpolate import interp2d


image = imread('../dcraw/baby.tiff')

# Report the image's width, height, and bits per pixel
height, width = image.shape[:2]  # Assumes the image is either grayscale or RGB
bits_per_pixel = image.dtype.itemsize * 8  # Byte size to bits
print(f"Width: {width} pixels, Height: {
      height} pixels, Bits Per Pixel: {bits_per_pixel}")

# Convert the image to a double-precision floating-point array
image_double = image.astype(np.float64)

# Optionally, you can print out some information about the conversion if necessary
print(f"Converted image dtype: {image_double.dtype}")

# Constants from the RAW image conversion
black = 0
white = 16383
multipliers = [1.628906, 1.000000, 1.386719, 1.000000]  # Not used in this part

# Linearize the image
linearized_image = (image_double - black) / (white - black)
linearized_image = np.clip(linearized_image, 0, 1)


# Assuming 'image' is your raw image loaded as a 2D numpy array
top_left_2x2 = linearized_image[0:2, 0:2]

# For now, let's print this out to see the raw values
print("Top-left 2x2 pixel values:\n", top_left_2x2)

# Initialize empty channels based on half the size, since RGGB takes 2x2 for a full color pixel
red_channel = np.zeros((height // 2, width // 2))
green_channel = np.zeros((height // 2, width // 2))
blue_channel = np.zeros((height // 2, width // 2))

# Extracting the channels based on RGGB pattern
red_channel = image[0::2, 0::2]
green_channel = (image[0::2, 1::2] + image[1::2, 0::2]) / 2
blue_channel = image[1::2, 1::2]


def white_world_balancing(red, green, blue):
    # Find the maximum value among all channels
    max_value = np.max([red.max(), green.max(), blue.max()])
    return red / max_value, green / max_value, blue / max_value


def gray_world_balancing(red, green, blue):
    # Calculate the average for each channel
    avg_r = np.mean(red)
    avg_g = np.mean(green)
    avg_b = np.mean(blue)
    avg_gray = (avg_r + avg_g + avg_b) / 3
    # Scale each channel by the global average divided by the channel average
    return (red * avg_gray / avg_r), (green * avg_gray / avg_g), (blue * avg_gray / avg_b)


def custom_white_balancing(red, green, blue, r_scale, g_scale, b_scale):
    # Apply the provided scale to each channel
    return red * r_scale, green * g_scale, blue * b_scale


# Multipliers from RAW conversion
r_scale, g_scale, b_scale = 1.628906, 1.000000, 1.386719

# Applying each white balancing method
red_ww, green_ww, blue_ww = white_world_balancing(
    red_channel, green_channel, blue_channel)
red_gw, green_gw, blue_gw = gray_world_balancing(
    red_channel, green_channel, blue_channel)
red_custom, green_custom, blue_custom = custom_white_balancing(
    red_channel, green_channel, blue_channel, r_scale, g_scale, b_scale)


def normalize_and_combine(red, green, blue):
    # Stack the channels along the third dimension
    combined_image = np.stack((red, green, blue), axis=-1)

    # Normalize or clip the image to be in the range [0, 1]
    combined_image = np.clip(combined_image, 0, 1)

    return combined_image


# Apply the white balancing techniques
# Note: Assuming red_channel, green_channel, blue_channel are correctly extracted from the original image
image_ww = normalize_and_combine(red_ww, green_ww, blue_ww)


def normalize_image(image):
    # Normalize the image to have values between 0 and 1
    max_val = np.max(image)
    min_val = np.min(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


# Combine the white balanced channels and normalize them
image_ww_combined = np.dstack((red_ww, green_ww, blue_ww))
image_ww_normalized = normalize_image(image_ww_combined)

image_gw_combined = np.dstack((red_gw, green_gw, blue_gw))
image_gw_normalized = normalize_image(image_gw_combined)

image_custom_combined = np.dstack((red_custom, green_custom, blue_custom))
image_custom_normalized = normalize_image(image_custom_combined)

# Visualization


def visualize_balanced_images(original, ww, gw, custom):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['Original', 'White World', 'Gray World', 'Custom White Balance']
    images = [original, ww, gw, custom]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Assuming 'linearized_image' needs to be normalized for display as well
linearized_normalized = normalize_image(linearized_image)

# Display the images
visualize_balanced_images(linearized_normalized, image_ww_normalized,
                          image_gw_normalized, image_custom_normalized)
