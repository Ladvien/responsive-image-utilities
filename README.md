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


### Reads
- [Adding Noise to Images](https://medium.com/@ms_somanna/guide-to-adding-noise-to-your-data-using-python-and-numpy-c8be815df524)
- [Image Scoring: Allocating Percentage Score to Images for Their Quality](https://medium.com/engineering-housing/image-scoring-allocating-percentage-score-to-images-for-their-quality-6169abbf850e)
- [Fine Grain Image Enhancer](https://huggingface.co/spaces/finegrain/finegrain-image-enhancer)
- [Pillow Docs](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html)
- [Image Resizing Algorithm Gallery](https://en.wikipedia.org/wiki/Comparison_gallery_of_image_scaling_algorithms)
- [Detecting Out of Focus Images](https://mathematica.stackexchange.com/questions/71726/how-can-i-detect-if-an-image-is-of-poor-quality)
- [Comprehensive Image Quality Detection Algorithms](https://medium.com/@jaikochhar06/how-to-evaluate-image-quality-in-python-a-comprehensive-guide-e486a0aa1f60)
- [Package Contain Algorithms Above](https://github.com/andrewekhalel/sewar)

### Building a Labeler
- [Flet](https://flet.dev/docs/cookbook/keyboard-shortcuts)

### Machine Learning Approaches
- [Reddit ML Quality Assessment](https://www.reddit.com/r/MachineLearning/comments/12v7jew/d_is_accurately_estimating_image_quality_even/)

#### Training Data
https://huggingface.co/datasets/laion/laion2B-multi-aesthetic
https://huggingface.co/datasets/laion/aesthetics_v2_4.75
https://huggingface.co/datasets/dclure/laion-aesthetics-12m-umap
https://huggingface.co/datasets/recastai/coyo-10m-aesthetic


## TODO:
- Rework the API to normalize image score.


## Done:
- Add license and attribution (https://opensource.stackexchange.com/questions/8096/how-can-i-correctly-apply-the-apache-2-0-licence-to-contributed-code-from-an-exi)

```md
### Summary Checklist:
1. **LICENSE file**: Include the full Apache 2.0 license text.
2. **NOTICE file**: Acknowledge the use of the original project, along with attribution and a URL if available.
3. **Copyright retention**: Retain copyright notices in the original files.
4. **Modifications**: Clearly indicate any modifications you've made.
5. **README/Documentation**: Add a mention of the Apache 2.0 code and attribution to the original authors in your documentation.

By following these steps, you'll ensure that you're correctly attributing the code and complying with the requirements of the Apache License 2.0.
```
