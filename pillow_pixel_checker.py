import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

ftypes = [
    ("Image", "*.jpg;*.jpeg;*.png;*.gif"),
    ("All files", "*.*")
]

def pick_color(event):
    x,y = event.x, event.y
    pixel_rgb = img_rgb[y,x]
    pixel_hsv = img_hsv[y,x]
    pixel_gray = img_gray[y,x]
    print('RGB : {:13} | HSV : {:13} | GRAY : {}'.format(str(pixel_rgb), str(pixel_hsv), str(pixel_gray)))

def press_enter(event) :
    exit()

if __name__=='__main__':
    root = tk.Tk()
    root.withdraw() # HIDE THE TKINTER GUI
    file_path = filedialog.askopenfilename(filetypes = ftypes)
    root.update()

    img = Image.open(file_path) # RGBA
    img_rgb = np.array(img.convert('RGB'))
    img_hsv = np.array(img.convert('HSV'))
    img_gray = np.array(img.convert('L'))

    width, height = img.size
    
    window = tk.Tk(className='Pillow Pixel Picker')
    window.bind('<Return>', press_enter)
    window.focus_force()
    canvas = tk.Canvas(window, width=width, height=height)
    canvas.pack()

    img_tk = ImageTk.PhotoImage(img, master=window)
    canvas.create_image(width // 2, height // 2, image=img_tk)
    
    canvas.bind("<Button-1>", pick_color)
    window.mainloop()
