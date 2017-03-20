# Design and Implementation of Algorithms to Rank Images of Varicosity
Environment: digital image processing, python, opencv, numpy
1. Segmented focal region and background with a modified 2D Otsu algorithm in YCgCr color space.
2. Extracted statistical information of lesion area include colors, gray values, standard variations and a series of texture vectors like contrast, entropy and LBP
3. Proposed a morphologic algorithm aimed to recognize raised veins with a special combination of light and shadow
4. Ranked images according to severity of varicosity with decision trees based on above morphologic and statistical data
