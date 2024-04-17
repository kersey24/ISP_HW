# Image Processing Pipeline README

## Overview

This script processes a RAW image file through several stages, including linearization, white balancing, demosaicing, color space correction, and final adjustments like brightness and gamma encoding. The purpose is to transform a RAW image into a high-quality image ready for viewing and analysis.

## Setup

To run this script, ensure you have the following Python libraries installed:

- `numpy`
- `matplotlib`
- `skimage`
- `colour_demosaicing`

You can install these packages using pip:

```bash
pip install numpy matplotlib scikit-image colour-demosaicing

Customization

To view different stages of the image processing:

Comment/Uncomment the plotting sections: Throughout the script, there are blocks of code that use matplotlib to display the image at various stages of processing. By default, these blocks are commented out. To view the results at any stage, uncomment these sections.

For example, to view the results after demosaicing, find the section labeled "Uncomment to plot demosaicing images" and uncomment it.
```
