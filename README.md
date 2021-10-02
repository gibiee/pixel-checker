# opencv-pixel-picker
You can check pixel value(BGR, HSV, etc.) on the image.  
Also you can add other color spaces by yourself.  
It must use GUI.

# Requirements
- Python3
- Tkinter
- OpenCV or Pillow

# Knowledge
OpenCV and Pillow have different value range in the specific color space.
||OpenCV|Pillow|
|:---:|:---:|:---:|
|**RGB**|[0, 0, 0] ~ [255, 255, 255]|[0, 0, 0] ~ [255, 255, 255]|
|**HSV**|[0, 0, 0] ~ [179, 255, 255]|[0, 0, 0] ~ [255, 255, 255]|


# Run
```python opencv_pixel_checker.py``` or ```python pillow_pixel_checker.py```

You can check pixel value on the image by mouse left click.  
Then, you can close the window by pressing enter key.

# Screenshot
- opencv-pixel-checker
![example1](https://user-images.githubusercontent.com/37574274/135706818-7561d06d-7303-43cc-a1c8-65577ac08ac5.png)
- pillow-pixel-checker
![example2](https://user-images.githubusercontent.com/37574274/135706819-ab84e6ea-ea4d-42b1-8945-714a5d7a15a5.png)
