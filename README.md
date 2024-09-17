# responsive-image-utilities
A module for creating responsive images, with a simple API.

## Description
- Image srcset tags are created
- Module can detect images resized too small, avoids it
- Outputs images in different formats
- Handles odd characters in file paths
- Folder paths can handle relative and absolute

### Inputs
- HTML
- Folder

```py
srcets = ImageFolder('./images').set_output_folder('./output')
```

### Reads
- [Adding Noise to Images](https://medium.com/@ms_somanna/guide-to-adding-noise-to-your-data-using-python-and-numpy-c8be815df524)
- [Image Scoring: Allocating Percentage Score to Images for Their Quality](https://medium.com/engineering-housing/image-scoring-allocating-percentage-score-to-images-for-their-quality-6169abbf850e)
- [Fine Grain Image Enhancer](https://huggingface.co/spaces/finegrain/finegrain-image-enhancer)
- [Pillow Docs](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html)
- [Image Resizing Algorithm Gallery](https://en.wikipedia.org/wiki/Comparison_gallery_of_image_scaling_algorithms)
- [Detecting Out of Focus Images](https://mathematica.stackexchange.com/questions/71726/how-can-i-detect-if-an-image-is-of-poor-quality)
- [Comprehensive Image Quality Detection Algorithms](https://medium.com/@jaikochhar06/how-to-evaluate-image-quality-in-python-a-comprehensive-guide-e486a0aa1f60)
- [Package Contain Algorithms Above](https://github.com/andrewekhalel/sewar)