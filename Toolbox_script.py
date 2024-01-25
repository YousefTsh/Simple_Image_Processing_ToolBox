# Yousef Taher Nageeb Shouman
# Sec2
# Computer Department
# Image Processing Tool Box


import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt


class ToolBox():
    def __init__(self,win):
        self.window =win
        self.window.configure(bg='#89B5AF')
        self.window.title('Yousef_Taher_Toolbox')
        self.window.geometry('950x650+520+30')
        self.window.resizable(True, True)
        #needed variables
        self.src = None
        self.x_zoom=[]
        self.y_zoom=[]
        self.pre_src = None
        self.img2 = None
        self.src_f=None
        self.out = None
        self.seg_dst=None
        self.p_s=[]
        self.p_d=[]

    #app
        Label(self.window, text='Image Tool Box', fg='#DED9C4',bg='#89B5AF', font=('Time New Roman', 16, 'bold')).place(x=430, y=0)
    #Tool frame
        # interface design
        self.tool_fram =Frame(self.window,bg='#D0CAB2',bd=3,relief="sunken")
        self.tool_fram.place(x=0,y=170,width=130,height=315)
        Label(self.tool_fram, text='',bd=10,relief="raised",bg='black').place(x=62, y=0, width=1, height=500)
        self.translation_b= Button(self.tool_fram,text='transl',bd=3,relief="raised",bg='gray',command=self.translation).place(x=0,y=0,width=60)
        self.rotation_b = Button(self.tool_fram, text='rotate', bd=3, relief="raised",bg='gray',command=self.rotate).place(x=65, y=0, width=60)
        Label(self.tool_fram, text='', bd=10, relief="raised", bg='black').place(x=0, y=32, width=500, height=1)
        self.skwing_b = Button(self.tool_fram, text='skewing', bd=3, relief="raised",bg='gray',command=self.skewing).place(x=0, y=34, width=60)
        self.zoom_b = Button(self.tool_fram, text='zoom', bd=3, relief="raised",bg='gray',command=self.zoom).place(x=65, y=34, width=60)
        self.flip_b = Button(self.tool_fram, text='flip', bd=3, relief="raised",bg='gray',command=self.flip).place(x=0, y=66, width=60)
        self.histeq_b = Button(self.tool_fram, text='HistEq', bd=3, relief="raised",bg='gray',command=self.histEQ).place(x=65, y=66, width=60)
        self.neg_b = Button(self.tool_fram, text='Negative', bd=3, relief="raised",bg='gray',command=self.negative).place(x=0, y=97, width=60)
        self.log_b = Button(self.tool_fram, text='Logrithm', bd=3, relief="raised",bg='gray',command=self.logarithmic).place(x=65, y=97, width=60)
        self.powl_b = Button(self.tool_fram, text='powlow', bd=3, relief="raised",bg='gray',command=self.powerlow).place(x=0, y=128, width=60)
        self.bitmap_b = Button(self.tool_fram, text='bitmap', bd=3, relief="raised",bg='gray',command=self.bitmap).place(x=65, y=128, width=60)
        self.Grayslice_b = Button(self.tool_fram, text='grayslice', bd=3, relief="raised",bg='gray',command=self.graylvlslice).place(x=0, y=160, width=60)
        self.blinding_b = Button(self.tool_fram, text='blinding', bd=3, relief="raised",bg='gray',command = self.blinding).place(x=65, y=160, width=60)
        self.smothing_b = Button(self.tool_fram, text='Smooth', bd=3, relief="raised",bg='gray',command = self.smoothing).place(x=0, y=190, width=60)
        self.median_f = Button(self.tool_fram, text='MedianF', bd=3, relief="raised",bg='gray',command = self.median_filter).place(x=65, y=190, width=60)
        self.Soble_f = Button(self.tool_fram, text='SobleF', bd=3, relief="raised",bg='gray',command = self.soble_filter).place(x=0, y=220, width=60)
        self.frequancy_D = Button(self.tool_fram, text='FDomain', bd=3, relief="raised",bg='gray',command = self.FDomain).place(x=65, y=220, width=60)
        self.histogram = Button(self.tool_fram, text='histogram', bd=3, relief="raised",bg='gray',command = self.histogram).place(x=0, y=250, width=60)
        self.segment = Button(self.tool_fram, text='segment', bd=3, relief="raised",bg='gray',command = self.segmentation).place(x=65, y=250, width=60)
        self.sharp = Button(self.tool_fram, text='sharp', bd=3, relief="raised",bg='gray',command = self.sharpin).place(x=0, y=280, width=60)
        self.sharp = Button(self.tool_fram, text='Prewitt', bd=3, relief="raised",bg='gray',command = self.prewitt_filter).place(x=65, y=280, width=60)

        # get the image
        self.open= Button(self.window,text='open image',bg='gray',bd=3,relief='raised',command=self.show).place(x=230,y=80,width=80,height=40)
        self.save = Button(self.window,text='save image',bg='gray',bd=3,relief='raised',command=self.save).place(x=330,y=80,width=80,height=40)
        self.back = Button(self.window,text='back',bg='gray',bd=3,relief='raised',command=self.back).place(x=430,y=80,width=80,height=40)
        self.rest = Button(self.window,text='rest',bg='gray',bd=3,relief='raised',command=self.rest).place(x=530,y=80,width=80,height=40)

    #function to to open the image you want to use it in the processing a dialog appears to select your image
    def show(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("All Files", ".png"),("All Files", ".jpg"),("All Files", ".jpeg"),('pngImages', '*.png'),('jpgimages','*.jpg'),('jpegimages','*.jpeg')])
        if self.file_path:
            self.src=cv2.imread(self.file_path,0)
            self.src_f=self.src
            cv2.namedWindow('Image',0)
            cv2.imshow('Image', self.src)

    #rest image and cancel all the effect made on it
    def rest(self):
        try:
            if self.file_path:
                self.src=cv2.imread(self.file_path,0)
                cv2.imshow('Image', self.src)
        except:
            pass

    #save the image after finishing the processing on it
    def save(self):
        if type(self.src) == np.ndarray:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",filetypes=[ ('pngImages', '.png'),('jpgimages', '.jpg'), ('jpegimages', '.jpeg'),("All Files", ".png"), ("All Files", ".jpg"), ("All Files", ".jpeg")])
            if file_path:
                print(file_path)
                cv2.imwrite(file_path,self.src)
        else:
            messagebox.showerror('Error', 'No image to Save', parent=self.window)

    # cancel the effect that was made on the image and go back to before this effect occurred
    def back(self):
        if type(self.pre_src) == np.ndarray:
            self.src = self.pre_src
            cv2.imshow('Image', self.src)

    # design frame of rotation operation
    def rotate(self):
        if type(self.src)==np.ndarray:
            self.pre_src=self.src
            self.slider_val = IntVar()
            self.slider_val.set(0)
            self.rotate_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.rotate_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.rotate_frame)
            Label(self.rotate_frame, text='Angle', bg='#D0CAB2').place(x=0, y=0)
            self.rot_slider = Scale(self.rotate_frame, from_=-180, to=180, orient=HORIZONTAL, length=180,command= lambda s : self.slider_val.set(int(s))).place(x=50,y=19)
            self.ok_rot= Button(self.rotate_frame, text='OK', bg='gray', bd=3, relief='raised',command=self.rot_ok).place(x=70, y=70)
            self.cancel_rot = Button(self.rotate_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.rotate_frame.destroy).place(x=130, y=70)


        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    # rotate the image according to the selected angle
    def rot_ok(self):
        self.pre_src=self.src
        self.src= cv2.resize(self.src,(self.src.shape[1],self.src.shape[1]))
        rows, cols = self.src.shape
        R = cv2.getRotationMatrix2D((rows // 2, cols // 2), self.slider_val.get(), 1)
        self.src = cv2.warpAffine(self.src, R, self.src.shape)
        cv2.imshow('Image', self.src)

    # Integrates two images after select second image first then
    # select the ratio of the first image and the ratio of the second image
    def blinding(self):
        def ok():
            if type(self.img2)==np.ndarray:
                if self.img1_r.get() +self.img2_r.get()==1:
                    for i in range(0, self.src.shape[0]):
                        for j in range(0, self.src.shape[1]):
                            self.src[i,j] = self.src[i,j]*self.img1_r.get() + self.img2[i,j]*self.img2_r.get()

                    cv2.imshow('Image',self.src)
                    cv2.destroyWindow('Image2')
                    self.pre_src=self.src_f
                else:
                    messagebox.showwarning('warning', 'sum of r1 , r2 must equal 1  ', parent=self.window)
            else:
                messagebox.showwarning('warning', 'select 2nd image first', parent=self.window)

        # frame design of blinding
        if type(self.src)==np.ndarray:
            self.img1_r= DoubleVar()
            self.img2_r = DoubleVar()
            self.blinding_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.blinding_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.blinding_frame)
            Label(self.blinding_frame, text='img1_r: ', bg='#D0CAB2').place(x=0, y=0)
            self.x = Spinbox(self.blinding_frame,textvariable=self.img1_r,from_=0.0, to_=0.9, increment = 0.1 ,width=5).place(x=50,y=0)
            Label(self.blinding_frame, text='img2_r: ', bg='#D0CAB2').place(x=0, y=30)
            self.y = Spinbox(self.blinding_frame, textvariable=self.img2_r,from_=0.0, to_=0.9,increment = 0.1 ,width=5).place(x=50, y=30)
            self.open_2 =  Button(self.blinding_frame, text='open', bg='gray', bd=3, relief='raised',command=self.open_blinding).place(x=10, y=65)
            self.ok= Button(self.blinding_frame, text='OK', bg='gray', bd=3, relief='raised',command=ok).place(x=60, y=65)
            self.cancel_translate = Button(self.blinding_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.blinding_frame.destroy).place(x=100, y=65)

        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    #fuction to open the second image in blinding function
    def open_blinding(self):
        file_path = filedialog.askopenfilename(filetypes=[("All Files", ".png"), ("All Files", ".jpg"), ("All Files", ".jpeg"), ('pngImages', '*.png'),('jpgimages', '*.jpg'), ('jpegimages', '*.jpeg')])
        if file_path:
            self.img2 = cv2.imread(file_path, 0)
            self.img2=cv2.resize(self.img2,(self.src.shape[1],self.src.shape[0]))
            cv2.namedWindow('Image2', 0)
            cv2.imshow('Image2', self.img2)

    # translation fram design
    def translation(self):
        if type(self.src)==np.ndarray:
            self.pre_src = self.src
            self.x_tv=IntVar()
            self.y_tv = IntVar()
            cv2.setMouseCallback('Image', self.mouse_Tfunc)
            self.translate_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.translate_frame.place(x=650, y=300, width=300, height=100)
            self.show_fram(self.translate_frame)
            Label(self.translate_frame, text='X: ', bg='#D0CAB2').place(x=0, y=0)
            self.x = Spinbox(self.translate_frame,textvariable=self.x_tv,from_=0, to_=500, width=5).place(x=10,y=0)
            Label(self.translate_frame, text='Y: ', bg='#D0CAB2').place(x=0, y=30)
            self.y = Spinbox(self.translate_frame, textvariable=self.y_tv,from_=0, to_=500, width=5).place(x=10, y=30)
            self.ok= Button(self.translate_frame, text='OK', bg='gray', bd=3, relief='raised',command=self.ok_translate).place(x=10, y=65)
            self.cancel_translate = Button(self.translate_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.translate_frame.destroy).place(x=50, y=65)

        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    # function get selected coordinate and translate image to it
    def ok_translate(self):
        Tm = np.float32([[1, 0, int(self.x_tv.get())], [0, 1, int(self.y_tv.get())]])
        self.src = cv2.warpAffine(self.src,Tm,self.src.shape)
        cv2.imshow('Image',self.src)


    #mouse function in translation to translate image to the selected location by the mouse
    def mouse_Tfunc(self,event,x,y,flag,pram):
        self.x_mouse=[]
        self.y_mouse=[]
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y)
            self.x_mouse.append(x)
            self.y_mouse.append(y)
        if self.x_mouse and self.y_mouse != []:
            Tm = np.float32([[1, 0, self.x_mouse[-1]], [0, 1, self.y_mouse[-1]]])
            print(Tm)
            self.src = cv2.warpAffine(self.src, Tm, self.src.shape)
            cv2.imshow('Image', self.src)

    def skewing(self):
        if type(self.src) == np.ndarray:
            self.pre_src = self.src
            self.p1_s = StringVar()
            self.p2_s = StringVar()
            self.p3_s = StringVar()
            self.p1_d = StringVar()
            self.p2_d = StringVar()
            self.p3_d = StringVar()

            self.Skew_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.Skew_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.Skew_frame)
            Label(self.Skew_frame, text='P1_s: ', bg='#D0CAB2').place(x=0, y=0)
            x1 = Entry(self.Skew_frame,textvariable=self.p1_s).place(x=50, y=0,width=50)
            Label(self.Skew_frame, text='P2_s: ', bg='#D0CAB2').place(x=0, y=20)
            x1 = Entry(self.Skew_frame, textvariable=self.p2_s).place(x=50, y=20,width=50)
            Label(self.Skew_frame, text='P3_s: ', bg='#D0CAB2').place(x=0, y=40)
            x1 = Entry(self.Skew_frame, textvariable=self.p3_s).place(x=50, y=40,width=50)
            Label(self.Skew_frame, text='P1_d: ', bg='#D0CAB2').place(x=100, y=0)
            x1 = Entry(self.Skew_frame, textvariable=self.p1_d).place(x=150, y=0, width=50)
            Label(self.Skew_frame, text='P2_d: ', bg='#D0CAB2').place(x=100, y=20)
            x1 = Entry(self.Skew_frame, textvariable=self.p2_d).place(x=150, y=20, width=50)
            Label(self.Skew_frame, text='P3_d: ', bg='#D0CAB2').place(x=100, y=40)
            x1 = Entry(self.Skew_frame, textvariable=self.p3_d).place(x=150, y=40, width=50)
            self.cancel_translate = Button(self.Skew_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.Skew_frame.destroy).place(x=50, y=65)
            cv2.setMouseCallback('Image', self.mouse_Skewingfunc)
        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)
    # function apply skewing after select 3 src points and 3 dst points
    def mouse_Skewingfunc(self,event,x,y,flag,pram):
        self.src=cv2.resize(self.src,(self.src.shape[1],self.src.shape[1]))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.p_s.append([x,y])
            self.p_d.append([x,y])
            if len(self.p_s)==3:
                self.p_s=self.p_s[0:3]
                print(self.p_s)
                self.p1_s.set(f'[{self.p_s[0][0]},{self.p_s[0][1]}]')
                self.p2_s.set(f'[{self.p_s[1][0]},{self.p_s[1][1]}]')
                self.p3_s.set(f'[{self.p_s[2][0]},{self.p_s[2][1]}]')
            if len(self.p_d) == 6:
                self.p_d = self.p_d[3:6]
                print(self.p_d)
                self.p1_d.set(f'[{self.p_d[0][0]},{self.p_d[0][1]}]')
                self.p2_d.set(f'[{self.p_d[1][0]},{self.p_d[1][1]}]')
                self.p3_d.set(f'[{self.p_d[2][0]},{self.p_d[2][1]}]')
                src_p = np.float32([self.p_s[0], self.p_s[1],self.p_s[2]])
                print(src_p)
                dst_p = np.float32([self.p_d[0], self.p_d[1],self.p_d[2]])
                SM = cv2.getAffineTransform(src_p,dst_p)
                self.src= cv2.warpAffine(self.src,SM,self.src.shape)
                cv2.imshow('Image', self.src)
                self.p_s=[]
                self.p_d=[]

    #zoom frame design
    def zoom(self):
        if type(self.src)==np.ndarray:
            cv2.setMouseCallback('Image', self.mouse_zoomfunc)
            self.pre_src = self.src
            self.x1_zoomv=IntVar()
            self.y1_zoomv = IntVar()
            self.x2_zoomv = IntVar()
            self.y2_zoomv = IntVar()

            self.zoom_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.zoom_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.zoom_frame)
            Label(self.zoom_frame, text='X1: ', bg='#D0CAB2').place(x=0, y=0)
            x1 = Spinbox(self.zoom_frame,textvariable=self.x1_zoomv,from_=0, to_=500, width=5).place(x=20,y=0)
            Label(self.zoom_frame, text='Y1: ', bg='#D0CAB2').place(x=100, y=0)
            y1 = Spinbox(self.zoom_frame, textvariable=self.y1_zoomv,from_=0, to_=500, width=5).place(x=120, y=0)

            Label(self.zoom_frame, text='X2: ', bg='#D0CAB2').place(x=0, y=30)
            x1 = Spinbox(self.zoom_frame, textvariable=self.x2_zoomv, from_=0, to_=500, width=5).place(x=20, y=30)
            Label(self.zoom_frame, text='Y2: ', bg='#D0CAB2').place(x=100, y=30)
            y1 = Spinbox(self.zoom_frame, textvariable=self.y2_zoomv, from_=0, to_=500, width=5).place(x=120, y=30)
            self.ok= Button(self.zoom_frame, text='OK', bg='gray', bd=3, relief='raised',command=self.ok_zoom).place(x=10, y=65)
            self.cancel_translate = Button(self.zoom_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.zoom_frame.destroy).place(x=50, y=65)
        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    # take the x1, y1, x2 ,y2 from the zoom frame and zoom the image acording to these values
    def ok_zoom(self):
        self.src = self.src[ self.y1_zoomv.get():self.y2_zoomv.get(),self.x1_zoomv.get():self.x2_zoomv.get()]
        self.src=cv2.resize(self.src,(self.pre_src.shape[1],self.pre_src.shape[0]))
        cv2.imshow('Image', self.src)

    # mouse fuction in zoom in the image by click on the 2 location you want, to create the part you want to zoom on it
    def mouse_zoomfunc(self,event,x,y,flag,pram):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x_zoom.append(x)
            self.y_zoom.append(y)
            print(x,y)
        if len(self.x_zoom)%2==0 and self.x_zoom !=[]:
            self.src=self.src[self.y_zoom[0]:self.y_zoom[1],self.x_zoom[0]:self.x_zoom[1]]
            self.x_zoom=[]
            self.y_zoom = []
            self.src = cv2.resize(self.src, (self.pre_src.shape[1], self.pre_src.shape[0]))
            cv2.imshow('Image',self.src)

    # fuction make flip operation based on the selected flip type
    def flip(self):
        self.pre_src = self.src
        def x_action():
            self.src = cv2.flip(self.src, 0)
            cv2.imshow('Image', self.src)
        def y_action():
            self.src = cv2.flip(self.src, 1)
            cv2.imshow('Image', self.src)
        def xy_action():
            self.src = cv2.flip(self.src, -1)
            cv2.imshow('Image', self.src)
        if type(self.src) == np.ndarray:
            self.flip_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.flip_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.flip_frame)
            flip_x = Button(self.flip_frame, text='Flip-X', bg='gray', bd=3, relief='raised',command=x_action).place(x=30, y=20)
            flip_y = Button(self.flip_frame, text='Flip-Y', bg='gray', bd=3, relief='raised',command=y_action).place(x=90, y=20)
            flip_xy = Button(self.flip_frame, text='Flip-XY', bg='gray', bd=3, relief='raised',command=xy_action).place(x=150, y=20)
            self.cancel_flip = Button(self.flip_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.flip_frame.destroy).place(x=50, y=65)
        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    # fuction apply the histogram equlization on the image
    def histEQ(self):
        self.pre_src = self.src
        if type(self.src) == np.ndarray:
            self.src=cv2.equalizeHist(self.src)
            cv2.imshow('Image',self.src)
        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    # fucntion to apply negative transformation on the image
    def negative(self):
        self.pre_src = self.src
        if type(self.src) == np.ndarray:
            self.src=cv2.bitwise_not(self.src)
            cv2.imshow('Image',self.src)
        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    # function to apply the logarithmic transformation on the image
    def logarithmic(self):
        self.pre_src = self.src
        if type(self.src) == np.ndarray:
            self.src= np.float32(self.src)
            c = 255 / (np.log(255+1))
            for i in range(0,self.src.shape[0]):
                for j in range(0,self.src.shape[1]):
                    self.src[i,j] = np.log(1+self.src[i,j])
            cv2.normalize(self.src,self.src,0,255,cv2.NORM_MINMAX)
            self.src=np.uint8(self.src)
            cv2.imshow('Image',self.src)

        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    # function to apply power transformation on the image according to the selected gamma
    def powerlow(self):
        if type(self.src) == np.ndarray:
            self.pre_src = self.src
            self.gamma = DoubleVar()
            self.powl_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.powl_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.powl_frame)
            Label(self.powl_frame, text='Gamma: ', bg='#D0CAB2').place(x=0, y=0)
            Gamma = Spinbox(self.powl_frame,textvariable=self.gamma,from_=0.00, to_=25.0, increment = 0.01 , width=5).place(x=50,y=0)
            self.ok = Button(self.powl_frame, text='OK', bg='gray', bd=3, relief='raised', command=self.ok_powl).place(x=10, y=65)
            self.cancel_translate = Button(self.powl_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.powl_frame.destroy).place(x=50, y=65)

        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    def ok_powl(self):
        self.src = np.float32(self.src)
        for i in range(0, self.src.shape[0]):
            for j in range(0, self.src.shape[1]):
                self.src[i, j] = pow(self.src[i, j], self.gamma.get())
        cv2.normalize(self.src, self.src, 0, 255, cv2.NORM_MINMAX)
        self.src = np.uint8(self.src)
        cv2.imshow('Image', self.src)

    # function to apply bitmap according to the selected plane you want to get
    def bitmap(self):
        self.pre_src = self.src
        if type(self.src) == np.ndarray:
            self.plan_no = StringVar()
            self.bitmap_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.bitmap_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.bitmap_frame)
            Label(self.bitmap_frame, text='plan no: ', bg='#D0CAB2').place(x=0, y=0)
            Gamma = Spinbox(self.bitmap_frame, textvariable=self.plan_no, from_=1, to_=8,width=5).place(x=50, y=0)
            self.ok = Button(self.bitmap_frame, text='OK', bg='gray', bd=3, relief='raised', command=self.ok_bitmap).place(x=10, y=65)
            self.cancel_translate = Button(self.bitmap_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.bitmap_frame.destroy).place(x=50, y=65)

        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    def ok_bitmap(self):
        dst=np.zeros(self.src.shape)
        plan_dect = {'1':1,'2':2,'3':4,'4':8,'5':16,'6':32,'7':64,'8':128}
        for i in range(0, self.src.shape[0]):
            for j in range(0, self.src.shape[1]):
                if self.src[i, j] & plan_dect[self.plan_no.get()]:
                    dst[i, j] = 255
                else:
                    dst[i, j] = 0
        cv2.imshow('Image', dst)
        self.src=dst
        cv2.imshow('Image',self.src)

    # gray lvl slicing function to highlight a specific range of intensities in the image
    def graylvlslice(self):
        if type(self.src) == np.ndarray:
            self.pre_src = self.src
            self.startlvl = IntVar()
            self.endlvl= IntVar()
            self.graylvl_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.graylvl_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.graylvl_frame)
            Label(self.graylvl_frame, text='start lvl: ', bg='#D0CAB2').place(x=0, y=0)
            g1 = Spinbox(self.graylvl_frame, textvariable=self.startlvl, from_=0, to_=255,width=5).place(x=50, y=0)
            Label(self.graylvl_frame, text='end lvl: ', bg='#D0CAB2').place(x=0, y=30)
            g2 = Spinbox(self.graylvl_frame, textvariable=self.endlvl, from_=0, to_=255, width=5).place(x=50, y=30)
            self.ok = Button(self.graylvl_frame, text='OK', bg='gray', bd=3, relief='raised', command=self.ok_graylvl).place(x=10, y=65)
            self.cancel_translate = Button(self.graylvl_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.graylvl_frame.destroy).place(x=50, y=65)

        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    def ok_graylvl(self):
        dst = np.zeros(self.src.shape)
        for i in range(0, self.src.shape[0]):
            for j in range(0, self.src.shape[1]):
                if self.src[i,j]>self.startlvl.get() and self.src[i,j]<self.endlvl.get():
                    dst[i,j]=255
                else:
                    dst[i,j]=0
        cv2.imshow("Image",dst)
        self.src=dst
        cv2.imshow("Image", self.src)

    #apply smoothing on the image (blur) by select on of the smooth filter from the opened frame to remove spark noise
    def smoothing(self):
        self.pre_src=self.src
        def avg():
            self.pre_src = self.src
            avg_mask = np.float32([[1,1,1],[1,1,1],[1,1,1]])
            avg_mask = avg_mask/9
            self.src=cv2.filter2D(self.src,cv2.CV_8UC1,avg_mask)
            cv2.imshow("Image",self.src)

        def gausian():
            self.pre_src = self.src
            self.pre_src=self.src
            gausian_mask = np.float32([[1,2,1],[2,4,2],[1,2,1]])
            gausian_mask = gausian_mask/16
            self.src=cv2.filter2D(self.src,cv2.CV_8UC1,gausian_mask)
            cv2.imshow("Image",self.src)

        def circulr():
            self.pre_src = self.src
            self.pre_src = self.src
            circulr_mask = np.float32([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])
            circulr_mask = circulr_mask/21
            self.src=cv2.filter2D(self.src,cv2.CV_8UC1,circulr_mask)
            cv2.imshow("Image",self.src)

        def cone():
            self.pre_src = self.src
            cone_mask = np.float32([[0,0,1,0,0],[0,2,2,2,0],[1,2,5,2,1],[0,2,2,2,0],[0,0,1,0,0]])
            cone_mask = cone_mask/25
            self.src=cv2.filter2D(self.src,cv2.CV_8UC1,cone_mask)
            cv2.imshow("Image",self.src)

        def pyram():
            self.pre_src = self.src
            pyram_mask = np.float32([[1,2,3,2,1],[2,4,6,4,2],[3,6,9,6,3],[2,4,6,4,2],[1,2,3,2,1]])
            pyram_mask = pyram_mask/81
            self.src=cv2.filter2D(self.src,cv2.CV_8UC1,pyram_mask)
            cv2.imshow("Image",self.src)

        if type(self.src) == np.ndarray:
            self.smooth_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.smooth_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.smooth_frame)
            avg = Button(self.smooth_frame, text='avgin', bg='gray', bd=3, relief='raised',command=avg).place(x=0, y=5)
            gausian = Button(self.smooth_frame, text='gausian', bg='gray', bd=3, relief='raised',command=gausian).place(x=50, y=5)
            circulr = Button(self.smooth_frame, text='circulr', bg='gray', bd=3, relief='raised',command=circulr).place(x=110, y=5)
            cone = Button(self.smooth_frame, text='cone', bg='gray', bd=3, relief='raised',command=cone).place(x=180, y=5)
            pyram = Button(self.smooth_frame, text='pyram', bg='gray', bd=3, relief='raised',command=pyram).place(x=0, y=40)
            self.cancel_translate = Button(self.smooth_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.smooth_frame.destroy).place(x=110, y=65)

        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    # apply median blur to remove salt and paper noise by select the number of element in the keranl to control the effect
    def median_filter(self):
        def median_ok():
            self.pre_src=self.src
            self.src = cv2.medianBlur(self.src,self.element_num.get())
            cv2.imshow("Image",self.src)

        if type(self.src) == np.ndarray:
            self.pre_src = self.src
            self.element_num = IntVar()
            self.median_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.median_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.median_frame)
            Label(self.median_frame, text='elements num : ', bg='#D0CAB2').place(x=0, y=0)
            g2 = Spinbox(self.median_frame, textvariable=self.element_num, from_=0, to_=255, width=5).place(x=100, y=0)
            self.cancel_translate = Button(self.median_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.median_frame.destroy).place(x=110, y=65)
            ok = Button(self.median_frame, text='OK', bg='gray', bd=3, relief='raised',command=median_ok).place(x=70, y=65)

        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    # function to apply soble filter (sharpening filter) to detect edges in the image you can
    # apply soblex to detect the vertical edges
    # apply sobley to detect the horizontal edges
    # apply soblexy to detect both
    def soble_filter(self):
        def SobleX():
            self.soblex = cv2.Sobel(self.pre_src,cv2.CV_16UC1,1,0,ksize=self.kernalSize.get())
            self.soblex=cv2.convertScaleAbs(self.soblex)
            cv2.namedWindow('soblex',0)
            cv2.imshow("soblex",self.soblex)
            self.src=self.soblex
        def SobleY():
            self.sobley = cv2.Sobel(self.pre_src,cv2.CV_16UC1,0,1,ksize=self.kernalSize.get())
            self.sobley=cv2.convertScaleAbs(self.sobley)
            cv2.namedWindow('sobley',0)
            cv2.imshow("sobley",self.sobley)
            self.src=self.sobley

        def SobleXY():
            self.soblex = cv2.Sobel(self.pre_src, cv2.CV_16UC1, 1, 0, ksize=self.kernalSize.get())
            self.soblex = cv2.convertScaleAbs(self.soblex)
            self.sobley = cv2.Sobel(self.pre_src,cv2.CV_16UC1,0,1,ksize=self.kernalSize.get())
            self.sobley=cv2.convertScaleAbs(self.sobley)
            self.soblexy=cv2.addWeighted(self.soblex,1,self.sobley,1,0)
            cv2.namedWindow('soblexy',0)
            cv2.imshow("soblexy",self.soblexy)
            self.src=self.soblexy

        def sobel_cancel():
            self.soble_frame.destroy()
            try:
                cv2.destroyWindow('soblex')
            except:
                pass
            try:
                cv2.destroyWindow('sobley')
            except:
                pass
            try:
                cv2.destroyWindow('soblexy')
            except:
                pass
            cv2.imshow('Image', self.src)
        if type(self.src) == np.ndarray:
            self.pre_src = self.src
            self.kernalSize=IntVar()
            self.soble_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.soble_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.soble_frame)
            Soble_x = Button(self.soble_frame, text='SobleX', bg='gray', bd=3, relief='raised',command=SobleX).place(x=0, y=0)
            Soble_Y = Button(self.soble_frame, text='SobleY', bg='gray', bd=3, relief='raised',command=SobleY).place(x=50, y=0)
            Soble_T = Button(self.soble_frame, text='SobleXY', bg='gray', bd=3, relief='raised',command=SobleXY).place(x=100, y=0)
            Label(self.soble_frame, text='Ksize : ', bg='#D0CAB2').place(x=0, y=40)
            g2 = Spinbox(self.soble_frame, textvariable=self.kernalSize, from_=0, to_=255, width=5).place(x=50, y=40)
            self.cancel_translate = Button(self.soble_frame, text='cancel', bg='gray', bd=3, relief='raised',command=sobel_cancel).place(x=110, y=65)

        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    # to apply the smooth or sharpening in the frequency domain
    # select from it the operation you want (smoothing, sharpening) after select one of them
    # the image converted to Frequency domain and a slider appear to select the cutoff frequency
    # that control the Impact degree of the smoothing or sharpening
    def FDomain(self):
        def lowpass():
            self.src_f = self.src
            planes = []
            m = cv2.getOptimalDFTSize(self.src_f.shape[0])
            n = cv2.getOptimalDFTSize(self.src_f.shape[1])
            padded = cv2.copyMakeBorder(self.src_f, 0, m - self.src_f.shape[0], 0, n - self.src_f.shape[1], 0)
            padded = np.float32(padded)
            cv2.normalize(padded, padded, 0, 1, cv2.NORM_MINMAX)
            planes.append(padded)
            planes.append(np.zeros(padded.shape, dtype=np.float32))
            complix = cv2.merge((planes[0], planes[1]))
            complix = cv2.dft(complix)
            planes[0], planes[1] = cv2.split(complix)
            FI = self.DFTshow(planes)
            cv2.namedWindow('before', 0)
            cv2.imshow('before', FI)
            complix = np.fft.fftshift(complix)
            planes[0], planes[1] = cv2.split(complix)
            FI = self.DFTshow(planes)
            cv2.namedWindow('after', 0)
            cv2.imshow('after', FI)
            Lfilter = np.zeros(complix.shape[0:2], dtype=np.float32)
            cv2.namedWindow('filter', 0)
            for i in range(0, Lfilter.shape[0]):
                for j in range(0, Lfilter.shape[1]):
                    z1 = i - Lfilter.shape[0] / 2
                    z2 = j - Lfilter.shape[1] / 2
                    if np.sqrt(pow(z1, 2) + pow(z2, 2)) > self.val.get():
                        Lfilter[i, j] = 0
                    else:
                        Lfilter[i, j] = 1

            cv2.imshow('filter', Lfilter)
            outR = cv2.multiply(planes[0],Lfilter)
            outI=cv2.multiply(planes[1],Lfilter)
            out_planes=[]
            out_planes.append(outR)
            out_planes.append(outI)
            out_complix = cv2.merge((out_planes[0],out_planes[1]))
            out_complix = np.fft.fftshift(out_complix)
            out_complix=cv2.idft(out_complix)
            out_planes[0],out_planes[1]=cv2.split(out_complix)
            self.out = cv2.magnitude(out_planes[0], out_planes[1])
            cv2.normalize(self.out,self.out,0,1,cv2.NORM_MINMAX)
            cv2.imshow('Image',self.out)

        self.val = DoubleVar()
        def f_smooth():
            self.pre_src=self.src
            lowpass()
            self.fdomain_slider = Scale(self.FDomain_frame, from_=0, to=200, orient=HORIZONTAL, length=200,command=lambda s: self.val.set(int(s))).place(x=30, y=29)
            ok = Button(self.FDomain_frame, text='ok', bg='gray', bd=3, relief='raised',command = lowpass).place(x=30, y=70)

        self.val1 = DoubleVar()
        def highpass():
            self.src_f = self.src
            planes = []
            m = cv2.getOptimalDFTSize(self.src_f.shape[0])
            n = cv2.getOptimalDFTSize(self.src_f.shape[1])
            padded = cv2.copyMakeBorder(self.src_f, 0, m - self.src_f.shape[0], 0, n - self.src_f.shape[1], 0)
            padded = np.float32(padded)
            cv2.normalize(padded, padded, 0, 1, cv2.NORM_MINMAX)
            planes.append(padded)
            planes.append(np.zeros(padded.shape, dtype=np.float32))
            complix = cv2.merge((planes[0], planes[1]))
            complix = cv2.dft(complix)
            planes[0], planes[1] = cv2.split(complix)
            FI = self.DFTshow(planes)
            cv2.namedWindow('before', 0)
            cv2.imshow('before', FI)
            complix = np.fft.fftshift(complix)
            planes[0], planes[1] = cv2.split(complix)
            FI = self.DFTshow(planes)
            cv2.namedWindow('after', 0)
            cv2.imshow('after', FI)
            Lfilter = np.zeros(complix.shape[0:2], dtype=np.float32)
            cv2.namedWindow('filter', 0)
            for i in range(0, Lfilter.shape[0]):
                for j in range(0, Lfilter.shape[1]):
                    z1 = i - Lfilter.shape[0] / 2
                    z2 = j - Lfilter.shape[1] / 2
                    if np.sqrt(pow(z1, 2) + pow(z2, 2)) < self.val1.get():
                        Lfilter[i, j] = 0
                    else:
                        Lfilter[i, j] = 1

            cv2.imshow('filter', Lfilter)
            outR = cv2.multiply(planes[0],Lfilter)
            outI=cv2.multiply(planes[1],Lfilter)
            out_planes=[]
            out_planes.append(outR)
            out_planes.append(outI)
            out_complix = cv2.merge((out_planes[0],out_planes[1]))
            out_complix = np.fft.fftshift(out_complix)
            out_complix=cv2.idft(out_complix)
            out_planes[0],out_planes[1]=cv2.split(out_complix)
            self.out = cv2.magnitude(out_planes[0], out_planes[1])
            cv2.normalize(self.out,self.out,0,1,cv2.NORM_MINMAX)
            cv2.imshow('Image',self.out)


        def f_sharpin():
            self.pre_src = self.src
            highpass()
            self.fdomain_slider = Scale(self.FDomain_frame, from_=0, to=200, orient=HORIZONTAL, length=200,command=lambda s: self.val1.set(int(s))).place(x=30, y=29)
            ok = Button(self.FDomain_frame, text='ok', bg='gray', bd=3, relief='raised', command=highpass).place(x=30,y=70)

        if type(self.src) == np.ndarray:
            self.FDomain_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.FDomain_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.FDomain_frame)
            Smooth = Button(self.FDomain_frame, text='Smooth', bg='gray', bd=3, relief='raised',command = f_smooth).place(x=50, y=0)
            Sharpien = Button(self.FDomain_frame, text='Sharpin', bg='gray', bd=3, relief='raised',command=f_sharpin).place(x=130, y=0)
            self.cancel_F = Button(self.FDomain_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.FDomain_cancel).place(x=110, y=70)

        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    def DFTshow(self,planes):
        mag = cv2.magnitude(planes[0],planes[1])
        mag = mag+1
        mag = np.log(mag)
        cv2.normalize(mag,mag ,0, 1, cv2.NORM_MINMAX)
        return mag
    def FDomain_cancel(self):
        self.FDomain_frame.destroy()
        if type(self.out)==np.ndarray:
            cv2.destroyWindow('before')
            cv2.destroyWindow('filter')
            cv2.destroyWindow('after')
            self.src=self.out
            cv2.imshow('Image',self.src)

    def sharpin(self):
        def ok():
            self.pre_src = self.src

            kernal = np.empty((self.shapekernal.get(), self.shapekernal.get()))
            for i in range(0, self.shapekernal.get()):
                for j in range(0, self.shapekernal.get()):
                    if i == self.shapekernal.get() // 2 and j == self.shapekernal.get() // 2:
                        kernal[i, j] = (self.shapekernal.get() * self.shapekernal.get())
                    else:
                       kernal[i, j] = -1
            kernal=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            kernal = np.float32(kernal)
            self.src = cv2.filter2D(self.src, -1, kernal)
            cv2.imshow("Image", self.src)
        if type(self.src)==np.ndarray:
            self.pre_src = self.src
            self.shapekernal = IntVar()
            self.sharp_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.sharp_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.sharp_frame)
            Label(self.sharp_frame, text='Ksize : ', bg='#D0CAB2').place(x=0, y=20)
            g2 = Spinbox(self.sharp_frame, textvariable=self.shapekernal, from_=0, to_=255, width=5).place(x=50, y=20)
            ok = Button(self.sharp_frame, text='ok', bg='gray', bd=3, relief='raised', command=ok).place(x=70, y=65)
            self.cancel_translate = Button(self.sharp_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.sharp_frame.destroy).place(x=110, y=65)
            # s=self.pre_src-self.src
            # cv2.imshow('im2',self.pre_src)
            # cv2.imshow('im3',s)
        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)


    # function to show the histogram of the image
    def histogram(self):
        if type(self.src)==np.ndarray:
            plt.hist(self.src.ravel(), bins=256, range=[0, 255])
            plt.show()
        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)


    #apply the segmentation by threshold or edge base methods
    def segmentation(self):
        self.segment_val=IntVar()
        def thersholding():
            self.seg_dst = np.empty(self.src.shape)
            for i in range(0,self.src.shape[0]):
                for j in range(0,self.src.shape[1]):
                    if self.src[i,j]>self.segment_val.get():
                        self.seg_dst[i,j]=255
                    else:
                        self.seg_dst[i,j]=0
            cv2.namedWindow('thershold segmentation',0)
            cv2.imshow('thershold segmentation',self.seg_dst)

        def thersholding_item():
            self.segment_slider = Scale(self.segment_frame, from_=0, to=220, orient=HORIZONTAL, length=200,command=lambda s: self.segment_val.set(int(s))).place(x=30, y=29)
            self.segment_ok = Button(self.segment_frame, text='ok', bg='gray', bd=3, relief='raised', command=thersholding).place(x=30,y=70)

        def edge_base():
            #kernal=[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,24,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]
            #[-1,-1,-1],[-1,8,-1],[-1,-1,-1]
            kernal = np.float32([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
            self.seg_dst=cv2.filter2D(self.src,-1,kernal)
            cv2.namedWindow('edgebase segmentation', 0)
            cv2.imshow('edgebase segmentation', self.seg_dst)

        if type(self.src) == np.ndarray:
            self.pre_src=self.src
            self.segment_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.segment_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.segment_frame)
            b1 = Button(self.segment_frame, text='Threshold', bg='gray', bd=3, relief='raised',command = thersholding_item).place(x=50, y=0)
            b2 = Button(self.segment_frame, text='EdgeBase', bg='gray', bd=3, relief='raised',command=edge_base).place(x=130, y=0)
            self.cancel_seg = Button(self.segment_frame, text='cancel', bg='gray', bd=3, relief='raised',command=self.segment_cancel).place(x=110, y=70)

        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    def segment_cancel(self):
        self.segment_frame.destroy()
        try:
            cv2.destroyWindow('thershold segmentation')
        except:
            pass
        try:
            cv2.destroyWindow('edgebase segmentation')
        except:
            pass
        if type(self.seg_dst)==np.ndarray:
            self.src=self.seg_dst
            cv2.imshow('Image',self.src)

    #sharpening filter used to detect edges of image
    def prewitt_filter(self):
        def prewittX():
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            self.prewittx = cv2.filter2D(self.pre_src,-1,kernelx)
            cv2.namedWindow('prewittx',0)
            cv2.imshow("prewittx",self.prewittx)
            self.src=self.prewittx
        def prewittY():
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            self.prewitty = cv2.filter2D(self.pre_src,-1,kernely)
            cv2.namedWindow('prewitty',0)
            cv2.imshow("prewitty",self.prewitty)
            self.src=self.prewitty

        def prewittXY():
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            self.prewittx = cv2.filter2D(self.pre_src, -1, kernelx)
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            self.prewitty = cv2.filter2D(self.pre_src, -1, kernely)
            self.prewittxy=cv2.addWeighted(self.prewittx,1,self.prewitty,1,0)
            cv2.namedWindow('prewittxy',0)
            cv2.imshow("prewittxy",self.prewittxy)
            self.src=self.prewittxy

        def prewitt_cancel():
            self.prewitt_frame.destroy()
            try:
                cv2.destroyWindow('prewittx')
            except:
                pass
            try:
                cv2.destroyWindow('prewitty')
            except:
                pass
            try:
                cv2.destroyWindow('prewittxy')
            except:
                pass
            cv2.imshow('Image', self.src)

        if type(self.src) == np.ndarray:
            self.pre_src = self.src
            self.prewitt_frame = Frame(self.window, bg='#D0CAB2', bd=3, relief="sunken")
            self.prewitt_frame.place(x=650, y=300, width=350, height=100)
            self.show_fram(self.prewitt_frame)
            Soble_x = Button(self.prewitt_frame, text='prewittX', bg='gray', bd=3, relief='raised',command=prewittX).place(x=0, y=0)
            Soble_Y = Button(self.prewitt_frame, text='prewittY', bg='gray', bd=3, relief='raised',command=prewittY).place(x=60, y=0)
            Soble_T = Button(self.prewitt_frame, text='prewittXY', bg='gray', bd=3, relief='raised',command=prewittXY).place(x=120, y=0)
            self.cancel_translate = Button(self.prewitt_frame, text='cancel', bg='gray', bd=3, relief='raised',command=prewitt_cancel).place(x=110, y=65)

        else:
            messagebox.showwarning('warning', 'select the image first', parent=self.window)

    #function to show a specefic frame when click on the button
    def show_fram(self, fram):
        fram.tkraise


if __name__ =='__main__':
    window = Tk()
    app=ToolBox(window)
    mainloop()
