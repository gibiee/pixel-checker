import tkinter as tk
from tkinter import filedialog
import cv2

ftypes = [
    ("Image", "*.jpg;*.jpeg;*.png;*.gif"),
    ("All files", "*.*")
]

def pick_color(event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = img[y,x]
        pixel_hsv = img_hsv[y,x]
        pixel_gray = img_gray[y,x]
        pixel_lab = img_lab[y,x]
        print('BGR : {:13} | HSV : {:13} | Lab : {:13} | GRAY : {}'.format(str(pixel), str(pixel_hsv), str(pixel_lab), str(pixel_gray)))

if __name__=='__main__':
    #OPEN DIALOG FOR READING THE IMAGE FILE
    root = tk.Tk()
    root.withdraw() # HIDE THE TKINTER GUI
    file_path = filedialog.askopenfilename(filetypes = ftypes)
    root.update()

    img = cv2.imread(file_path) # BGR
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    cv2.imshow("OpenCV Pixel Picker", img)

    # CALLBACK FUNCTION
    cv2.setMouseCallback("OpenCV Pixel Picker", pick_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
