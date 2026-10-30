# Handcrafted Image Baseline

This baseline uses only locally available Fakeddit images and avoids pretrained visual models.

Image loading uses Pillow for decoding, RGB conversion, and resizing. Feature computation is manual with numpy.

Features:

- original width
- original height
- original aspect ratio
- per-channel RGB mean
- per-channel RGB standard deviation
- luminance mean and standard deviation
- Sobel edge magnitude mean and standard deviation
- normalized RGB histogram bins

Luminance is computed as:

```text
Y = 0.299R + 0.587G + 0.114B
```

The Sobel edge magnitude is:

```text
sqrt(Gx^2 + Gy^2)
```

Features are standardized with training-set mean and standard deviation:

```text
z = (x - mean_train) / std_train
```

The classifier is the existing manually implemented binary logistic regression.
