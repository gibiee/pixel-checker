import gradio as gr
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from collections import Counter

img = Image.open('color_map.png').convert('RGB')
img_array = np.array(img)

mask = img.convert('L')
mask_bool = np.where(np.array(mask) > 128, True, False)

rgb_pixels = img_array[mask_bool]

r_counter = Counter(rgb_pixels[:, 0].tolist())
g_counter = Counter(rgb_pixels[:, 1].tolist())
b_counter = Counter(rgb_pixels[:, 2].tolist())

dict = {
    'value': list(range(256)),
    'R_count': [r_counter.get(i, 0) for i in range(256)],
    'G_count': [g_counter.get(i, 0) for i in range(256)],
    'B_count': [b_counter.get(i, 0) for i in range(256)]
}
df = pd.DataFrame(dict)
df

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
def make_tuple(pixel) :
    if isinstance(pixel, np.ndarray) :
        return tuple(map(int, pixel))
    else :
        return pixel

def check_pixel_value(pil_img: Image.Image, evt: gr.SelectData):
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
    return [make_tuple(pixel) for pixel in pixels]

def extract_from_editor(editor) :
    img = editor['background']
    mask = editor['layers'][0].split()[3]
    if len(np.unique(mask)) > 1 : img.putalpha(mask)
    return img

def analysis_pixel_value(img_rgba: Image.Image) :
    img_rgb, img_hsv, mask = np.array(img.convert('RGB')), np.array(img.convert('HSV')), np.array(mask)
    mask_bool = np.where(mask > 128, True, False)
    
    rgb_pixels = img_rgb[mask_bool]
    hsv_pixels = img_hsv[mask_bool]



with gr.Blocks(js=js_func) as demo :

    with gr.Tab("Check pixel value") :
        with gr.Row() :
            input_img = gr.Image(sources=['upload'], type='pil', image_mode='RGB', scale=3,
                                 show_label=False, show_download_button=False, show_share_button=False)

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

        gr.Examples(examples=['color_map.png'], inputs=input_img)
        input_img.select(check_pixel_value, inputs=[input_img], outputs=[pil_rgb, pil_hsv, pil_grayscale, opencv_bgr, opencv_rgb, opencv_hsv, opencv_hsv_full, opencv_lab, opencv_grayscale])


    with gr.Tab("Distribution Analysis") :
        with gr.Row() :
            input_img = gr.ImageEditor(sources=['upload'], type='pil', image_mode='RGBA', layers=False,
                                       show_label=False, show_download_button=False, show_share_button=False,
                                       transforms=[], brush=gr.Brush(colors=['AAAAAA'], default_color='AAAAAA', color_mode='fixed'))
            
            cut_img = gr.Image(type='pil', image_mode='RGBA', show_label=False, show_download_button=False, show_share_button=False)

        btn = gr.Button('Analysis', variant='primary')
        btn.click(fn=extract_from_editor, inputs=[input_img], outputs=[cut_img])

    with gr.Row() :
        gr.BarPlot(df, x='value', y='R_count', x_lim=[0, 255], x_bin=10)
        gr.BarPlot(df, x='value', y='G_count', x_lim=[0, 255], x_bin=10)
        gr.BarPlot(df, x='value', y='B_count', x_lim=[0, 255], x_bin=10)



demo.launch()