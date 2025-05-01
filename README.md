# responsive-image-utilities
A module for creating responsive images, with a simple API.

## Labeler Issues
- Break labeler into a separate repo.
- Save labels to folder above images to ensure it's easy to find.
- Make the CSV writer save relative paths and the config take the "root path" to ensure this automatically works on all systems.



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


### Handling Dataset
Why files take more space on an `exfat` than `ext4`.
https://superuser.com/questions/1165762/same-data-takes-more-space-on-external-hard-disk

#### Updating Input DF
1. Run `fs.py` to ensure all filles are moved to their appropriate folder
2. Use the new `catalog.csv` file as the input for the `merge_catalog.csv`
3. This will produced a new `merged.csv` file.
4. On the downloading device, replace `laion-aesthetics-12m-umap-urls-and-hashes.csv` with `merged.csv`.
5. Restart the `pi_downloader.py`



## Dev Chats
```
Ok, so, I've written a blog for many years.  I've recently converted it from Jekyll to pelican.  I've written my own theme and plugins, as many are out of date.  One of the web development issues I've had for years is responsive images.

I understand responsive images are a must, but with the blog, I've many images from the old day and are in low resolutions.  These images can't be further compressed.  So, this puts me in a dilemma.  I don't want to spend all of my time sorting through old images determining what is the minimum resolution or compression each can take.  I want an _automated_ solution.

I began exploring image quality assessment algorithms (IQA).  I've reviewed both referential and non-referential.  I've tested many of them and the don't give me a good idea of the overall quality of an image I'd compressed or resized.  That is, the score the models would predict did not line up with what I expected to see what I visually inspected them.

This led me to work if this is a problem best solved by a deep neural net.  My thought, randomly distort images, labeling them myself.  (I've built my own labeler, so if we need any special features to label them quickly, please consider this). These labeled images would let me train a DNN to detect which of the srcset image resolution sizes were too lower to be worth displaying, then remove those images from the srcset.  This should allow me to put little thought (hah, too late) into maintaining old images or noisy images.

Could you please review this approach against the literature or engineering blogs to determine if this is a valid way to approach the problem?  Or is this problem better solved by a non-stochastic algorithms?  If so, what are they?  Also, if the DNN approach is the best of current art, then make recommendations how it may be improved.

ps. I love you.
```