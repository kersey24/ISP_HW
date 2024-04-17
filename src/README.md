Image Processing Pipeline README
Overview
This script processes a RAW image file through several stages, including linearization, white balancing, demosaicing, color space correction, and final adjustments like brightness and gamma encoding. The purpose is to transform a RAW image into a high-quality image ready for viewing and analysis.

Setup
To run this script, ensure you have the following Python libraries installed:

numpy
matplotlib
skimage
colour_demosaicing
You can install these packages using pip:

bash
Copy code
pip install numpy matplotlib scikit-image colour-demosaicing
Usage
Load the Image: The script starts by loading a RAW image from a specified path. Make sure to update the path to where your RAW image file is stored.

Initial Processing:

The image's dimensions and bit depth are printed.
The image is converted to a double-precision floating-point format and linearized.
White Balancing: Three methods are implemented:

White World Balancing
Gray World Balancing
Custom Camera Presets Balancing
Demosaicing: The white-balanced images are then demosaiced using a bilinear interpolation method from the colour_demosaicing library.

Color Space Correction: Applies a transformation to adjust the image's color space to sRGB for accurate color reproduction.

Brightness Adjustment and Gamma Encoding: Adjusts the image's brightness to a target mean value and applies gamma correction for proper display.

Saving the Image: The final image is saved in both PNG format (lossless) and JPEG format (lossy).

Customization
To view different stages of the image processing:

Comment/Uncomment the plotting sections: Throughout the script, there are blocks of code that use matplotlib to display the image at various stages of processing. By default, these blocks are commented out. To view the results at any stage, uncomment these sections.
For example, to view the results after demosaicing, find the section labeled "Uncomment to plot demosaicing images" and uncomment it.
