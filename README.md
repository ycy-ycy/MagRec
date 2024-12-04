# Universal Reconstruction of Complex Magnetic Profiles

This repository contains the code accompanying the paper:

**"Universal Reconstruction of Complex Magnetic Profiles with Minimum Prior Assumptions"**

*Authors*: Changyu Yao, Yinyao Shi, Ji-In Jung, Zoltán Váci, Yizhou Wang, Zhongyuan Liu, Yue Yu, Chuanwei Zhang, Sonia Tikoo-Schantz, Chong Zu

*Published*: December 2, 2024

*Currently available on*: [arXiv](https://arxiv.org/abs/2411.18882)

## Requirements

- CUDA-enabled GPU
- Required Python packages:
  - NumPy
  - CuPy
  - Matplotlib
  - Numba
  - OpenCV
  - Pillow
  - psutil
  - Tkinter (for GUI components)
  - Jupyter Notebook (for `.ipynb` files)

## Usage

*Coordinates*: We use right-handed Cartesian coordinates with the origin at the bottom-left corner of the magnetic field image. All pixels in the magnetic field image have positive x-coordinates and positive y-coordinates, with a z-coordinate of zero. $\hat{z}$ direction is defined as $\hat{x} \times \hat{y}$.

*Display*: In all plots, the positive x-direction is displayed from left to right, and the positive y-direction is displayed from bottom to top. The z-coordinate remains constant across one plot.

*Data format*: Input magnetic data should be provided as three separate 2D arrays (i.e. `Bx.npy`, `By.npy`, and `Bz.npy`). Use `dataReshaper.py` to convert a 3D array into these 2D arrays. The first index corresponds to the x-coordinate, and the second index corresponds to the y-coordinate. Increasing index values indicate increasing x and y coordinates. When using region allocation from `png` files or upload region from `npy` files, ensure that the physical scale of the magnetic source image matches the scale of the magnetic field image in both width and height.

*Instructions*: The `?` buttons provide detailed instructions and data format requirements for each step.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite the original paper:

```
@article{yao2024universal,
  title={Universal Reconstruction of Complex Magnetic Profiles with Minimum Prior Assumptions},
  author={Yao, Changyu and Shi, Yinyao and Jung, Ji-In and Vaci, Zoltan and Wang, Yizhou and Liu, Zhongyuan and Yu, Yue and Zhang, Chuanwei and Tikoo-Schantz, Sonia and Zu, Chong},
  journal={arXiv preprint arXiv:2411.18882},
  year={2024}
}
```

## Acknowledgements

We thank the contributors and maintainers of the open-source libraries used in this project, including NumPy, CuPy, Matplotlib, Numba, OpenCV, Pillow, and others.

For any questions or issues, please open an issue on this repository or contact the authors directly.

