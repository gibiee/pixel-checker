import gradio as gr
from PIL import Image
import cv2
import numpy as np

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

def print_pixel(pixel) :
    if isinstance(pixel, np.ndarray) :
        return tuple(map(int, pixel))
    else :
        return pixel

with gr.Blocks(js=js_func) as demo:

    with gr.Row():
        color_map = gr.Image(value='color_map.png', scale=3, type='pil',
                             interactive=False, show_label=False, show_download_button=False, show_share_button=False)

        with gr.Column(scale=1):
            pil_rgb = gr.Textbox(label='PIL RGB', interactive=False)
            pil_hsv = gr.Textbox(label='PIL HSV', interactive=False)
            pil_grayscale = gr.Textbox(label='PIL Grayscale', interactive=False)
        
        with gr.Column(scale=1):
            opencv_rgb = gr.Textbox(label='OpenCV RGB', interactive=False)
            opencv_bgr = gr.Textbox(label='OpenCV BGR', interactive=False)
            opencv_hsv = gr.Textbox(label='OpenCV HSV', interactive=False)
            opencv_hsv_full = gr.Textbox(label='OpenCV HSV_FULL', interactive=False)
            opencv_lab = gr.Textbox(label='OpenCV Lab', interactive=False)
            opencv_grayscale = gr.Textbox(label='OpenCV Grayscale', interactive=False)

    def get_select_coords(pil_img: Image, evt: gr.SelectData):
        pointX, pointY = evt.index

        pil_rgb = np.array(pil_img.convert('RGB'))[pointY, pointX]
        pil_hsv = np.array(pil_img.convert('HSV'))[pointY, pointX]
        pil_grayscale = np.array(pil_img.convert('L'))[pointY, pointX]

        img_rgb = np.array(pil_img.convert('RGB'))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        opencv_bgr = img_bgr[pointY, pointX]
        opencv_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)[pointY, pointX]
        opencv_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[pointY, pointX]
        opencv_hsv_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV_FULL)[pointY, pointX]
        opencv_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)[pointY, pointX]
        opencv_grayscale = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[pointY, pointX]

        pixels = [pil_rgb, pil_hsv, pil_grayscale, opencv_bgr, opencv_rgb, opencv_hsv, opencv_hsv_full, opencv_lab, opencv_grayscale]
        return [print_pixel(pixel) for pixel in pixels]
        
    color_map.select(get_select_coords, inputs=[color_map], outputs=[pil_rgb, pil_hsv, pil_grayscale, opencv_bgr, opencv_rgb, opencv_hsv, opencv_hsv_full, opencv_lab, opencv_grayscale])

demo.launch()