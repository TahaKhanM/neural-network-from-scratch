# MNIST data

These are the original unsigned-byte IDX tensors already present in the repository: 60,000 training images and labels and 10,000 test images and labels. Images are 28 × 28 grayscale pixels; labels are digits 0–9. MNIST is attributed to Yann LeCun, Corinna Cortes and Christopher Burges. Duplicate nested copies were removed in the 2026 revision; the four canonical files are unchanged.

The loader checks the unsigned-byte type, dimensions, exact payload length, image shape, label count and label range. It does not fetch data or use the test labels during training.
