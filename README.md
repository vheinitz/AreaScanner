# AreaScanner — stitch & identify a scene from images 📷

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An attempt to write an **area scanner** that stitches images together and
identifies the scene using **feature detectors** (OpenCV). Includes both a
C++ (Qt/OpenCV) core and a Python experiment.

## Idea

Capture overlapping images of a surface, detect and match local features
between adjacent frames, stitch them into one mosaic, and use the detected
features to recognise what the scene shows.

## Contents

| File | Purpose |
|------|---------|
| `main.cpp` | C++ entry point, OpenCV feature pipeline |
| `areascanner.py` | Python experiment with the same idea |
| `imagetool.pro` | qmake project for the C++ part |

## Requirements

- C++: Qt + OpenCV
- Python: OpenCV

## Build / run

```bash
# C++
qmake && make

# Python
python areascanner.py
```

## License

[MIT](LICENSE) © Valentin Heinitz
