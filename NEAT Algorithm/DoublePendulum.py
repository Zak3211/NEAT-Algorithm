import math

class doublePendulum:
    def __init__(self):
        self.boundary = 70

        #constants
        self.g = -9.81
        self.L1 = 100
        self.L2 = 100
        self.m1 = 100
        self.m2 = 100
        self.dt = 0.025
        self.width =1000
        self.by = 250

        #Angle Parameters
        self.theta1 = math.pi
        self.theta2 = math.pi
        self.theta1_ = 0
        self.theta2_ = 0
        self.theta1__ = 0
        self.theta2__ = 0

        #Base Parameters:
        self.xb = 500
        self.xb_ = 0
        self.xb__ = 0

        #Coordinate Parameters (Latent)
        self.x1 = self.xb + self.L1*math.sin(self.theta1)
        self.x2 = self.x1 + self.L2*math.sin(self.theta2)
        self.y1 =  self.by - self.L1*math.cos(self.theta1)
        self.y2 = self.y1 - self.L2*math.cos(self.theta2)

    
    def get_theta1__(self):
        self.theta1__ =  -(self.m1 + self.m2)*self.xb__*math.cos(self.theta1) 
        self.theta1__ -=  self.m2*self.L2*self.theta2__*math.cos(self.theta1 - self.theta2) 
        self.theta1__ -= self.m2*self.L2*(self.theta2_*self.theta2_)*math.sin(self.theta1 - self.theta2) 
        self.theta1__ -= self.g*(self.m1+self.m2)*math.sin(self.theta1)
        self.theta1__ /= (self.m1 + self.m2)*self.L1
    
    def get_theta2__(self):
        self.theta2__ = -self.g*math.sin(self.theta2)
        self.theta2__ -= self.xb__*math.cos(self.theta2)
        self.theta2__ += self.L1*self.theta1_*self.theta1_*math.sin(self.theta1 - self.theta2)
        self.theta2__ -= self.L1*self.theta1__*math.cos(self.theta1 - self.theta2)
        self.theta2__ /= self.L2
    
    def update(self):
        self.xb__ = min(self.xb__, 250)

        self.xb_ += self.xb__*self.dt
        self.xb += self.xb_*self.dt
        self.xb_ *= 0.9

        self.xb = max(self.xb, self.boundary)
        self.xb = min(self.xb, self.width - self.boundary)
        if self.xb == self.boundary or self.xb == self.width - self.boundary:
            self.xb__ = 0
         
        self.get_theta1__()
        self.get_theta2__()

        self.theta1__ = min(self.theta1__, 3)
        self.theta1__ = max(self.theta1__, -3)
        self.theta2__ = min(self.theta2__, 3)
        self.theta2__ = max(self.theta2__, -3)

        self.theta1_ += self.theta1__*self.dt
        self.theta2_ += self.theta2__*self.dt

        self.theta1 += self.theta1_*self.dt
        self.theta2 += self.theta2_*self.dt

        self.x1 = self.xb + self.L1*math.sin(self.theta1)
        self.x2 = self.x1 + self.L2*math.sin(self.theta2)
        self.y1 =  self.by - self.L1*math.cos(self.theta1)
        self.y2 = self.y1 - self.L2*math.cos(self.theta2)

        self.theta1_ *= 0.999
        self.theta2_ *= 0.999

        self.xb__ *= 0.99
        

import tkinter as tk
def displayPendulum():
    
    if True:
        window = tk.Tk()
        canvas = tk.Canvas(window, bg = 'black', highlightthickness= 0)
        screen_width = 1000
        screen_height = 500
        canvas.config(width = screen_width, height = screen_height)
        canvas.pack()

        by = 250
        p = doublePendulum()
        x1 = p.xb + p.L1*math.sin(p.theta1)
        y1 =  by - p.L1*math.cos(p.theta1)

        x2 = x1 + p.L2*math.sin(p.theta2)
        y2 = y1 - p.L2*math.cos(p.theta2)

        line1 = canvas.create_line(p.xb, by, x1, y1, fill = 'white', width = 2)
        point1 = canvas.create_oval(x1 - 10, y1 - 10, x1 + 10, y1 + 10, fill='white')

        line2 = canvas.create_line(x1, y1, x2, y2, fill = 'white', width = 2)
        point2 = canvas.create_oval(x2 - 10, y2 - 10, x2 + 10, y2 + 10, fill='white')      
        
        base = canvas.create_line(p.xb-20, by, p.xb+20, by, fill = 'gray', width = 5)
    
    def move_left(event):
        p.xb__ = -30
    def move_right(event):
        p.xb__ = 30

    window.bind('<Left>', move_left)
    window.bind('<Right>', move_right)

    def game_loop():
        p.update()

        x1 = p.xb + p.L1*math.sin(p.theta1)
        y1 = by - p.L1*math.cos(p.theta1)

        x2 = x1 + p.L2*math.sin(p.theta2)
        y2 = y1 - p.L2*math.cos(p.theta2)

        canvas.coords(line1, [p.xb, by, x1, y1])
        canvas.coords(point1, [x1 - 10, y1 - 10, x1 + 10, y1 + 10])

        canvas.coords(line2, [x1, y1, x2, y2])
        canvas.coords(point2, [x2 - 10, y2 - 10, x2 + 10, y2 + 10])  

        canvas.coords(base, [p.xb-20, by, p.xb+20, by])
        canvas.after(2, game_loop)
    
    game_loop()
    window.mainloop()

if __name__ == '__main__':
    displayPendulum()