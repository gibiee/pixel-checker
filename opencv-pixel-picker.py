import cv2
import tkinter as tk
from tkinter import filedialog

ftypes = [
    ("Image", "*.jpg;*.jpeg;*.png;*.gif"),
    ("All files", "*.*")
]

def pick_color(event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = img[y,x]
        pixel_hsv = img_hsv[y,x]
        print('BGR : {:15} |   HSV : {}'.format(str(pixel), str(pixel_hsv)))

if __name__=='__main__':
    # OPEN DIALOG FOR READING THE IMAGE FILE
    root = tk.Tk()
    root.withdraw() # HIDE THE TKINTER GUI
    file_path = filedialog.askopenfilename(filetypes = ftypes)
    root.update()

    img = cv2.imread(file_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.imshow("pixel-picker", img)

    # CALLBACK FUNCTION
    cv2.setMouseCallback("pixel-picker", pick_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
