# Apples

A demo of some computer vision and neural network capabilities. Uses OpenCV and Detectron2.

## Installation

The project consists of two parts that can be installed and used separately. Additionally a prebuilt project is available in pages branch.

### Python

The Python part is used for neural network training and export. Note that these instructions have only been tested with Micromamba on Ubuntu 20.04, other distros should work, Miniconda may have problems though.

1. Create and activate the environment

    ```bash
    micromamba env create -f environment.yml
    micromamba activate apples
    ```

1. Clone Detectron2 and some requirements

    ```bash
    cd ..
    git clone https://github.com/facebookresearch/mobile-vision
    git clone https://github.com/facebookresearch/detectron2
    git clone https://github.com/facebookresearch/d2go
    ```

1. Install Detectron2 and some requirements one by one

    ```bash
    cd mobile-vision
    pip install -e .
    cd ../detectron2
    pip install -e .
    cd ../d2go
    pip install -e .
    cd ../apples
    ```

1. Done, the project is ready to use, see the Python modules for further configuration

    ```bash
    # use this for training
    python -m py.runner
    # or this for annotations
    python -m py.annotate
    ```

### JavaScript

The JavaScript part is used for OpenCV and UI. Note that we use Volta for Node management but other methods should do.

1. Install all requirements

    ```bash
    npm ci
    ```

1. Yup, that's it, the project is ready to use, note that the npm scripts effectively execute the bash ones

    ```bash
    # use this for development
    npm run parcel-start
    # or this for building
    npm run parcel-build
    ```

## License

For licensing information see [LICENSE](./LICENSE.md).
