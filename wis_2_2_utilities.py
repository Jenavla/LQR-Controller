# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:32:57 2023

@author: Rik
"""
import numpy as np
import pychrono as chrono
import pychrono.irrlicht as irr
import math as m
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import glob
import shutil
import csv
import sys


frame_folder = './frames'
filename_temp = frame_folder+'/img_temp_'



def createFrameFolder():
  #this function creates a folder for the frames if none excists
  isExist = os.path.exists(frame_folder)

  if isExist:
    files = glob.glob('/frames')
    for f in files:
        os.remove(f)
  else:
    # Create a new directory because it does not exist 
    os.makedirs(frame_folder)
    print("The new directory for frames is created!")
    
    
def createGIF(filename_gif,frame_duration):
  #this function creates a GIF file from available frames and then deletes the frames.
  print('Creating GIF, please wait...')    

  frames = [Image.open(image) for image in glob.glob("./frames/*.JPG")]
  frame_one = frames[0]
  frame_one.save(filename_gif, format="GIF", append_images=frames[1:],
                 save_all=True, duration=frame_duration, loop=0, optimize = False)


  print('Created GIF, cleaning folder...')      

  for filename in os.listdir(frame_folder):
      file_path = os.path.join(frame_folder, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))

  print('Cleaned folders, all done!')    
  

    
class simulation():
  def __init__(self,model,timestep=0.001):
    # Initialise visualisation
    self.model = model
    self.title = model.title
    self.vis = irr.ChVisualSystemIrrlicht()
    self.vis.AttachSystem(self.model.system)
    self.vis.SetWindowSize(1024,768)
    self.vis.SetWindowTitle(self.title)
    self.vis.Initialize()
    self.vis.AddSkyBox()
    self.vis.AddCamera(model.camara_vector)
    self.vis.AddTypicalLights()

    #set observation filter
    self.state_filter=0
    self.log_estimate = False

    # Simulation parameters
    self.time=0
    self.timestep = timestep
    self.steps = 0
    self.render_steps = 20
    self.frame_duration = self.timestep*self.render_steps
    self.total_duration = 0
    self.max_duration = 600
    self.render_frame = 0
    self.filename = self.title
    self.data_mode = 'end' #set to end to only write at the end
    self.GIF_toggle = True #True creates GIF images.
    self.data = []
    
    #cost parameters
    self.costset = False
    self.costQ = 0 #the cost for the position space deviation
    self.costR = 0 #the cost for the input.
    self.costR_offset = 0 #put in an ofset for constant input
    self.cost_state = 0 #tracking total state cost
    self.cost_input = 0 #tracking total input cost
    
    self.observed = 0
    self.u_input = 0
    
    #delete previous data
    file_path = './'+ self.filename+'.csv'
    
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('No previous data csv file')
    
    # maak folder voor frames    
    createFrameFolder()
    

    
  def observe(self):
    self.observed = self.model.getState()
    return self.observed
    
  def refreshTime(self):
    print_str = 'time is ' + f'{self.time:.3f}'
    sys.stdout.write('\r'+print_str)
    sys.stdout.flush()
  
  def setCost(self, Q = 1, Q_offset = 0, R = 1, R_offset = 0, input_size = 1):
    state_size = len(self.model.getState())
    
    if isinstance(Q, int) or isinstance(Q, float):
      self.costQ = np.diag(np.array([Q]*state_size))
    elif isinstance(Q, list):
      self.costQ = np.diag(np.array([Q]))
    else:
      self.costQ = Q
    
    if isinstance(R, int) or isinstance(R, float):
      self.costR = np.array([R]*input_size)
    elif isinstance(R, list):
      self.costR = np.diag(np.array(R))
    else:
      self.costR = R  
      
    if isinstance(Q_offset , int) or isinstance(Q_offset, float):
      self.costQ_offset = np.array([0]*state_size)
      self.costQ_offset[0] = Q_offset
    elif isinstance(Q_offset, list):
      self.costQ_offset  = np.array([Q_offset])  
    
    if isinstance(R_offset , int) or isinstance(R_offset, float):
      self.costR_offset = np.array([R_offset]*input_size)
    elif isinstance(R_offset, list):
      self.costR_offset = np.array([R_offset])
    
    
    
  def control(self,u):
    if isinstance(u, int) or isinstance(u, float):
      self.u_input = np.array([u])
      self.model.updateActuator1(u)
    else:
      try:
        u = u.flatten()
        self.u_input = u
        self.model.updateActuator1(u[0])
        if len(u) > 1:
          self.model.updateActuator2(u[1])
      except:
        try:
          u = np.asarray(u)
          self.u_input = u
          self.model.updateActuator1(u[0])
          if len(u) > 1:
            self.model.updateActuator2(u[1])
        except:
          print('Wrong variable for feedback, check output of controller!')


  def step(self):
    self.vis.BeginScene() 
    self.vis.Render()
    self.vis.EndScene()
    self.time+=self.timestep
    self.steps+=1
    
    self.model.updateSystem(timestep=self.timestep)
    
    
    if (self.steps % self.render_steps == 0):
      if self.GIF_toggle:
        self.vis.WriteImageToFile(filename_temp+str(self.render_frame).zfill(5)+'.jpg')

        self.render_frame += 1
        
  
  def logEstimate(self, estimate):
   try: 
    self.log_estimate = True
    self.state_estimate = np.array(estimate).flatten().tolist()     
   except:
    pass
  
  def log(self):
    self.cost_state += self.timestep*((self.observed-self.costQ_offset) @ self.costQ @ (self.observed-self.costQ_offset))
    



    calc_part = (self.u_input-self.costR_offset) @ self.costR
    if calc_part.size == 1:
      try:
        calc_part = calc_part[0]
      except:
        pass
      self.cost_input += self.timestep*(calc_part * (self.u_input-self.costR_offset)[0])
    else:
      self.cost_input += self.timestep*(calc_part @ (self.u_input-self.costR_offset))
      

    data_row = [self.time,self.cost_state,self.cost_input] + self.observed.tolist() + self.u_input.tolist()
    
    if self.log_estimate:
      data_row += self.state_estimate
    
    
    
          
    if self.data_mode == 'end':
      self.data.append(data_row)
      

    else:      
      try:
          f = open(self.filename+'.csv', 'a', newline='')
          # create the csv writer
          writer = csv.writer(f)
      
          # write a row to the csv file
          
          writer.writerow(data_row)
          # close the file
          f.close()
          pass
      except:
          pass
    
  def writeData(self):
    if self.data_mode == 'end':
      try:
        f = open(self.filename+'.csv', 'a', newline='')
        # create the csv writer
        writer = csv.writer(f)
    
        # write a row to the csv file
        for data_row in self.data:
          writer.writerow(data_row)
        # close the file
        f.close()
        pass
      except:
        print('unable to write data, try changing data mode')
    
  
  
    if self.GIF_toggle:
      createGIF(self.filename+'.gif',self.frame_duration)
  

        