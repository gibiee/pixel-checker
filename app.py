import glob
import gradio as gr
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from collections import Counter

img = Image.open('samples/color_map.png').convert('RGBA')
img_array = np.array(img)
img_array.shape

mask = img.convert('L')
mask_bool = np.where(np.array(mask) > 128, True, False)
img_array[mask_bool].shape

r_pixels = img_array[:,:,0]
Counter(r_pixels.flatten().tolist())
np.unique(img_array[:, :, 3])

def check_pixel_value(img_pil_rgba: Image.Image, evt: gr.SelectData):
    pointX, pointY = evt.index

    pixel_pil_rgba = np.array(img_pil_rgba)[pointY, pointX]
    pixel_pil_hsv = np.array(img_pil_rgba.convert('HSV'))[pointY, pointX]
    pixel_pil_grayscale = np.array(img_pil_rgba.convert('L'))[pointY, pointX]

    img_cv_rgba = np.array(img_pil_rgba)
    img_cv_bgra = cv2.cvtColor(img_cv_rgba, cv2.COLOR_RGBA2BGRA)
    pixel_cv_bgra = img_cv_bgra[pointY, pointX]
    pixel_cv_rgba = cv2.cvtColor(img_cv_bgra, cv2.COLOR_BGRA2RGBA)[pointY, pointX]
    pixel_cv_hsv = cv2.cvtColor(img_cv_bgra, cv2.COLOR_BGR2HSV)[pointY, pointX]
    pixel_cv_hsv_full = cv2.cvtColor(img_cv_bgra, cv2.COLOR_BGR2HSV_FULL)[pointY, pointX]
    pixel_cv_lab = cv2.cvtColor(img_cv_bgra, cv2.COLOR_BGR2Lab)[pointY, pointX]
    pixel_cv_grayscale = cv2.cvtColor(img_cv_bgra, cv2.COLOR_BGRA2GRAY)[pointY, pointX]

    pixels = [pixel_pil_rgba, pixel_pil_hsv, pixel_pil_grayscale, pixel_cv_bgra, pixel_cv_rgba, pixel_cv_hsv, pixel_cv_hsv_full, pixel_cv_lab, pixel_cv_grayscale]
    for i, pixel in enumerate(pixels):
        if isinstance(pixel, np.ndarray) :
            pixels[i] = tuple(map(int, pixel))
    return pixels

def check_pixel_distribution(img_pil_rgba, mask_pil=None):
    if mask_pil is None:
        rgba_array = np.array(img_pil_rgba.convert('RGBA'))
        r_pixels, g_pixels, b_pixels, a_pixels = rgba_array[:,:,0], rgba_array[:,:,1], rgba_array[:,:,2], rgba_array[:,:,3]
    else:
        mask_bool = np.where(np.array(mask_pil.convert('L')) > 128, True, False)
        rgba_array = np.array(img_pil_rgba.convert('RGBA'))[mask_bool]
        r_pixels, g_pixels, b_pixels, a_pixels = rgba_array[:,0], rgba_array[:,1], rgba_array[:,2], rgba_array[:,3]
    
    r_counter = Counter(r_pixels.flatten().tolist())
    g_counter = Counter(g_pixels.flatten().tolist())
    b_counter = Counter(b_pixels.flatten().tolist())
    a_counter = Counter(a_pixels.flatten().tolist())

    img_pil_hsv = img_pil_rgba.convert('HSV')
    hsv_array = np.array(img_pil_hsv)
    h_pixels, s_pixels, v_pixels = hsv_array[:,:,0], hsv_array[:,:,1], hsv_array[:,:,2]
    h_counter = Counter(h_pixels.flatten().tolist())
    s_counter = Counter(s_pixels.flatten().tolist())
    v_counter = Counter(v_pixels.flatten().tolist())

    dict = {
        'value': list(range(256)),
        'R_count': [r_counter.get(i, 0) for i in range(256)],
        'G_count': [g_counter.get(i, 0) for i in range(256)],
        'B_count': [b_counter.get(i, 0) for i in range(256)],
        'A_count': [a_counter.get(i, 0) for i in range(256)],
        'H_count': [h_counter.get(i, 0) for i in range(256)],
        'S_count': [s_counter.get(i, 0) for i in range(256)],
        'V_count': [v_counter.get(i, 0) for i in range(256)]
    }
    df = pd.DataFrame(dict)
    return [df] * 7

def extract_from_editor(input_img, editor) :
    cut_img_array = np.array(input_img.convert('RGBA'))
    mask = editor['layers'][0].split()[3]
    mask_array = np.array(mask.convert('L'))
    cut_img_array[:, :, 3] = mask_array
    Image.fromarray(cut_img_array).save('cut_image_alpha.png')
    # 지금 문제가 있는 이유 : mask는 True/False로 binary 값으로 봐야하는데, alpha 값으로 적용하니 오류가 발생함
    return Image.fromarray(cut_img_array), mask

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
with gr.Blocks(js=js_func) as demo :

    with gr.Tab("Check Pixel-Value") :
        with gr.Row() :
            with gr.Column(scale=3):
                input_img = gr.Image(sources=['upload'], type='pil', image_mode='RGBA', scale=3, show_label=False)
                gr.Examples(examples=sorted(glob.glob('samples/*')), inputs=input_img)

            with gr.Column(scale=1):
                pil_rgb = gr.Textbox(label='PIL RGBA', interactive=False)
                pil_hsv = gr.Textbox(label='PIL HSV', interactive=False)
                pil_grayscale = gr.Textbox(label='PIL Grayscale', interactive=False)
            
            with gr.Column(scale=1):
                cv_rgba = gr.Textbox(label='OpenCV RGBA', interactive=False)
                cv_bgra = gr.Textbox(label='OpenCV BGRA', interactive=False)
                cv_hsv = gr.Textbox(label='OpenCV HSV', interactive=False)
                cv_hsv_full = gr.Textbox(label='OpenCV HSV_FULL', interactive=False)
                cv_lab = gr.Textbox(label='OpenCV Lab', interactive=False)
                cv_grayscale = gr.Textbox(label='OpenCV Grayscale', interactive=False)

        input_img.select(check_pixel_value, inputs=[input_img], outputs=[pil_rgb, pil_hsv, pil_grayscale, cv_bgra, cv_rgba, cv_hsv, cv_hsv_full, cv_lab, cv_grayscale])


    with gr.Tab("Check Pixel-Distribution") :
        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(sources=['upload'], type='pil', image_mode='RGBA', scale=3, show_label=False)
                gr.Examples(examples=sorted(glob.glob('samples/*')), inputs=input_img)

            with gr.Column(scale=3):
                with gr.Row():
                    r_barplot = gr.BarPlot(x='value', y='R_count', x_lim=[0, 255], x_bin=16)
                    g_barplot = gr.BarPlot(x='value', y='G_count', x_lim=[0, 255], x_bin=16)
                    b_barplot = gr.BarPlot(x='value', y='B_count', x_lim=[0, 255], x_bin=16)
                    a_barplot = gr.BarPlot(x='value', y='A_count', x_lim=[0, 255], x_bin=16)
                with gr.Row():
                    h_barplot = gr.BarPlot(x='value', y='H_count', x_lim=[0, 255], x_bin=16)
                    s_barplot = gr.BarPlot(x='value', y='S_count', x_lim=[0, 255], x_bin=16)
                    v_barplot = gr.BarPlot(x='value', y='V_count', x_lim=[0, 255], x_bin=16)

        input_img.change(fn=check_pixel_distribution, inputs=[input_img], outputs=[r_barplot, g_barplot, b_barplot, a_barplot, h_barplot, s_barplot, v_barplot])


    with gr.Tab("Check Pixel-Distribution by Mask") :
        gr.Markdown("gr.ImageEditor에서 알파 채널이 무시되는 이슈가 있기 때문에, gr.Image를 통해 이미지를 업로드합니다.")
        with gr.Row() :
            with gr.Column(scale=1):
                input_img = gr.Image(sources=['upload'], type='pil', image_mode='RGBA', show_label=False)
                input_editor = gr.ImageEditor(sources=['upload'], type='pil', image_mode='RGBA', layers=False, show_label=False, interactive=True)
                
                btn = gr.Button('Check', variant='primary')
                cut_img = gr.Image(type='pil', image_mode='RGBA', show_label=False, show_download_button=False, show_share_button=False, interactive=False)
                mask_img = gr.Image(type='pil', image_mode='L', show_label=False, show_download_button=False, show_share_button=False, interactive=False)
                gr.Examples(examples=sorted(glob.glob('samples/*')), inputs=input_img)

                input_img.change(fn=lambda x:x, inputs=[input_img], outputs=[input_editor])

            with gr.Column(scale=2):
                with gr.Row():
                    r_barplot = gr.BarPlot(x='value', y='R_count', x_lim=[0, 255], x_bin=16)
                    g_barplot = gr.BarPlot(x='value', y='G_count', x_lim=[0, 255], x_bin=16)
                    b_barplot = gr.BarPlot(x='value', y='B_count', x_lim=[0, 255], x_bin=16)
                    a_barplot = gr.BarPlot(x='value', y='A_count', x_lim=[0, 255], x_bin=16)
                with gr.Row():
                    h_barplot = gr.BarPlot(x='value', y='H_count', x_lim=[0, 255], x_bin=16)
                    s_barplot = gr.BarPlot(x='value', y='S_count', x_lim=[0, 255], x_bin=16)
                    v_barplot = gr.BarPlot(x='value', y='V_count', x_lim=[0, 255], x_bin=16)
        
        btn.click(fn=extract_from_editor, inputs=[input_img, input_editor], outputs=[cut_img, mask_img]).success(fn=check_pixel_distribution, inputs=[input_img, mask_img], outputs=[r_barplot, g_barplot, b_barplot, a_barplot, h_barplot, s_barplot, v_barplot])

demo.launch()
