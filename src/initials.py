from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear


image = imread('../dcraw/baby.tiff')

# Report the image's width, height, and bits per pixel
height, width = image.shape[:2]  # Assumes the image is either grayscale or RGB
bits_per_pixel = image.dtype.itemsize * 8  # Byte size to bits
print(f"Width: {width} pixels, Height: {
      height} pixels, Bits Per Pixel: {bits_per_pixel}")

# Convert the image to a double-precision floating-point array
image_double = image.astype(np.float64)

print(f"Converted image dtype: {image_double.dtype}")

# Constants from RAW image conversion
black = 0
white = 16383
multipliers = [1.628906, 1.000000, 1.386719, 1.000000]  # Not used in this part

# Linearize
linearized_image = (image_double - black) / (white - black)
linearized_image = np.clip(linearized_image, 0, 1)

top_left_2x2 = linearized_image[0:2, 0:2]

print("Top-left 2x2 pixel values:\n", top_left_2x2)


def apply_white_world(image):
    copy = image.copy()

    max_red = np.max(image[0::2, 0::2])
    max_green = max([np.max(image[0::2, 1::2]), np.max(image[1::2, 0::2])])
    max_blue = np.max(image[1::2, 1::2])

    copy[0::2, 0::2] /= max_red
    copy[0::2, 1::2] /= max_green
    copy[1::2, 0::2] /= max_green
    copy[1::2, 1::2] /= max_blue

    return copy


def apply_gray_world(image):
    copy = image.copy()
    mean_red = np.mean(image[0::2, 0::2])
    mean_green = np.mean((image[0::2, 1::2] + image[1::2, 0::2]) / 2)
    mean_blue = np.mean(image[1::2, 1::2])
    total_avg = (mean_red + mean_green + mean_blue) / 3
    copy[0::2, 0::2] *= total_avg/mean_red
    copy[0::2, 1::2] *= total_avg/mean_green
    copy[1::2, 0::2] *= total_avg/mean_green
    copy[1::2, 1::2] *= total_avg/mean_blue

    return copy


def apply_camera_presets(image, multipliers):
    copy = image.copy()

    copy[0::2, 0::2] *= multipliers[0]
    copy[1::2, 1::2] *= multipliers[2]
    print(copy.shape)
    return copy


white_world_balanced = apply_white_world(linearized_image)
gray_world_balanced = apply_gray_world(linearized_image)
camera_presets_balanced = apply_camera_presets(linearized_image, multipliers)


def display(image):
    return np.clip(image * 5, 0, 1)

# Uncomment to plot white balancing images

# fig, ax = plt.subplots(1, 4, figsize=(15, 5))
# ax[0].imshow(display(linearized_image), cmap='gray')
# ax[0].set_title('Original')
# ax[1].imshow(display(white_world_balanced), cmap='gray')
# ax[1].set_title('White World Balanced')
# ax[2].imshow(display(gray_world_balanced), cmap='gray')
# ax[2].set_title('Gray World Balanced')
# ax[3].imshow(display(camera_presets_balanced), cmap='gray')
# ax[3].set_title('Camera Presets Balanced')
# plt.savefig('white_balancing.png')
# plt.show()


ww_demosaicing = demosaicing_CFA_Bayer_bilinear(white_world_balanced, 'RGGB')
gw_demosaicing = demosaicing_CFA_Bayer_bilinear(gray_world_balanced, 'RGGB')
cp_demosaicing = demosaicing_CFA_Bayer_bilinear(
    camera_presets_balanced, 'RGGB')

# Uncomment to plot demosaicing images

# fig, ax = plt.subplots(1, 4, figsize=(15, 5))
# ax[0].imshow(display(linearized_image), cmap='gray')
# ax[0].set_title('Original Demosaiced')
# ax[1].imshow(display(ww_demosaicing), cmap='gray')
# ax[1].set_title('White World Balanced Demosaiced')
# ax[2].imshow(display(gw_demosaicing), cmap='gray')
# ax[2].set_title('Gray World Balanced Demosaiced')
# ax[3].imshow(display(cp_demosaicing), cmap='gray')
# ax[3].set_title('Camera Presets Balanced Demosaiced')
# plt.savefig('demosaicing.png')
# plt.show()

# BEST DEMOSAICED IMAGE
demosaiced_image = cp_demosaicing


# The given 1x9 vector
vector_MXYZ_to_cam = np.array(
    [6988, -1384, -714, -5631, 13410, 2447, -1485, 2204, 7318])

# Reshape and scale
MXYZ_to_cam = vector_MXYZ_to_cam.reshape((3, 3)) / 10000.0

# Given MsRGB_to_XYZ
MsRGB_to_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

# Compute MsRGB_to_cam
MsRGB_to_cam = np.dot(MXYZ_to_cam, MsRGB_to_XYZ)

# Normalize the matrix so that each row sums to 1
MsRGB_to_cam = MsRGB_to_cam / MsRGB_to_cam.sum(axis=1)[:, np.newaxis]

# Compute the inverse of MsRGB_to_cam
MsRGB_to_cam_inv = np.linalg.inv(MsRGB_to_cam)


# Function to apply color space transformation to the image
def apply_color_space_correction(image, transformation_matrix):

    flat_image = image.reshape((-1, 3))

    corrected_colors = np.dot(flat_image, transformation_matrix.T)

    corrected_image = corrected_colors.reshape(image.shape)
    # Clip values to be within a valid range
    corrected_image = np.clip(corrected_image, 0, 1)
    return corrected_image


corrected_image = apply_color_space_correction(
    demosaiced_image, MsRGB_to_cam_inv)

# Plot the color corrected image

# Plotting the corrected image
# plt.imshow(display(corrected_image))
# plt.title('Color Space Corrected Image')
# plt.savefig('color_corrected_image.png')
# plt.show()


# Function to adjust brightness
def adjust_brightness(image, target_mean=0.25):

    gray_image = rgb2gray(image)
    current_mean = np.mean(gray_image)
    scale_factor = target_mean / current_mean
    # Brighten the image by scaling

    brightened_image = image * scale_factor

    # Clip values to be in the range [0, 1]
    brightened_image = np.clip(brightened_image, 0, 1)
    return brightened_image

# Function for gamma encoding


def gamma_encode(image):
    # Define the sRGB gamma encoding function
    def srgb_gamma(c):
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return 1.055 * (c ** (1/2.4)) - 0.055

    # Apply the gamma function to each channel
    gamma_encoded_image = np.vectorize(srgb_gamma)(image)
    return gamma_encoded_image


# Adjust brightness
brightened_image = adjust_brightness(corrected_image, target_mean=0.25)

# Apply gamma encoding
gamma_encoded_image = gamma_encode(brightened_image)
final_save = (gamma_encoded_image * 255).astype(np.uint8)
print("corrected type", final_save.dtype)

# Display the final image


# plt.imshow(gamma_encoded_image)
# plt.title('Gamma Encoded and Brightness Adjusted Image')
# plt.savefig('Gamma Encoded and Brightness Adjusted Image.png')
# plt.show()


# Save the final image


# imsave('final_image.png', final_save)
# imsave('final_image_30%.jpg', final_save, quality=30)
# print("Image saved")
