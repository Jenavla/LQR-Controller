# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:32:57 2023

@author: Rik
"""
import numpy as np
import pychrono as chrono
# pychrono.irrlicht as irr
import math as m
#from PIL import Image, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
#import os
#import glob
#import shutil

#repeated parameter
rho_wood = 700 #kg/m3
rho_iron = 7870 #kg/m3

###############################################################################

def shiftAngle(angle):
    if angle < -m.pi:
        angle+=2*m.pi
    elif angle > m.pi:
        angle-=2*m.pi
    return angle

class MySpringForce(chrono.ForceFunctor):
    def __init__(self,spring_coef = 1,rest_length = 0,damping_coef = 1):
        super(MySpringForce, self).__init__()
        self.spring_coef = spring_coef
        self.rest_length = rest_length
        self.damping_coef = damping_coef
        

        
    
    def evaluate(self,         #
                 time,         # current time
                 rest_length,  # undeformed length
                 length,       # current length
                 vel,          # current velocity (positive when extending)
                 link):        # associated link
        force = -self.spring_coef * (length - self.rest_length) - self.damping_coef * vel
        return force
      
###############################################################################      
      
class mass_spring():
    def __init__(self,body_size1=0.2,spring_coef1=100,rest_length1=0,damping_coef1=0.0, second_body=False, body_size2=0.2,spring_coef2=10,rest_length2=0,damping_coef2=0.0):
        self.system = chrono.ChSystemNSC()
        self.title = ('mass_spring_system')
        
        # Set gravity
        self.system.Set_G_acc(chrono.ChVectorD(0, 0, 0))
        self.camara_vector = chrono.ChVectorD(-0.0, 4, 6)

        self.ground = chrono.ChBody()
        self.system.AddBody(self.ground)
        self.ground.SetIdentifier(-1)
        self.ground.SetBodyFixed(True)
        self.ground.SetCollide(False)
        
        self.body_1_size = body_size1

        sph_1 = chrono.ChSphereShape(0.1)
        self.ground.AddVisualShape(sph_1, chrono.ChFrameD(chrono.ChVectorD(0, 0, 0)))

        self.body_1 = chrono.ChBodyEasyBox(self.body_1_size, self.body_1_size, self.body_1_size, rho_wood, True, False)
        self.system.AddBody(self.body_1)
        self.body_1.SetPos(chrono.ChVectorD(0, -1, 0))


        # Attach a visualization asset.
        box_1 = chrono.ChBoxShape(self.body_1_size, self.body_1_size, self.body_1_size)
        box_1.SetColor(chrono.ChColor(0.6, 0, 0))
        self.body_1.AddVisualShape(box_1)

        # Create a ChForce object
        self.control_force = chrono.ChForce()

        # Add the force to the body
        self.body_1.AddForce(self.control_force)

        self.control_force.SetDir(chrono.ChVectorD(0, 1, 0))  # Force vector (global frame)
        self.control_force.SetMforce(0)

        # Create the spring between body_1 and ground. The spring end points are
        # specified in the body relative frames.
        self.spring_force1 = MySpringForce(spring_coef1, rest_length1, damping_coef1)
        self.spring_1 = chrono.ChLinkTSDA()
        self.spring_1.Initialize(self.body_1, self.ground, True, chrono.ChVectorD(0, 0, 0), chrono.ChVectorD(0, 0, 0))
        self.spring_1.RegisterForceFunctor(self.spring_force1)
        self.system.AddLink(self.spring_1)
        
        # Attach a visualization asset.
        self.spring_1.AddVisualShape(chrono.ChSpringShape(0.05, 80, 15))
        
        if second_body:
          self.body_2_size = body_size2

          self.body_2 = chrono.ChBodyEasyBox(self.body_2_size, self.body_2_size, self.body_2_size, rho_wood, True, False)
          self.system.AddBody(self.body_2)
          self.body_2.SetPos(chrono.ChVectorD(0, -2, 0))
          
          # Attach a visualization asset.
          box_2 = chrono.ChBoxShape(self.body_2_size, self.body_2_size, self.body_2_size)
          box_2.SetColor(chrono.ChColor(0.0, 0.6, 0))
          self.body_2.AddVisualShape(box_2)
          
          # Create the spring between body_2 and body_1. The spring end points are
          # specified in the body relative frames.
          self.spring_force2 = MySpringForce(spring_coef2, rest_length2, damping_coef2)
          self.spring_2 = chrono.ChLinkTSDA()
          self.spring_2.Initialize(self.body_2, self.body_1, True, chrono.ChVectorD(0, 0, 0), chrono.ChVectorD(0, 0, 0))
          self.spring_2.RegisterForceFunctor(self.spring_force2)
          self.system.AddLink(self.spring_2)
          
          # Attach a visualization asset.
          self.spring_2.AddVisualShape(chrono.ChSpringShape(0.05, 80, 15))
        
        self.position = 0
        self.velocity = 0
        self.second_body = second_body
        
        if self.second_body:
          self.position2 = 0
          self.velocity2 = 0
          
            
    def getState(self):
        #return the current state
        if self.second_body:
          return np.array([self.position,self.velocity,self.position2,self.velocity2])
        else:
          return np.array([self.position,self.velocity])
      
    def updateSystem(self, timestep = 1e-2):
        self.system.DoStepDynamics(timestep)
        self.position = self.body_1.GetPos().y
        self.velocity = self.body_1.GetPos_dt().y
        if self.second_body:
          self.position2 = self.body_2.GetPos().y
          self.velocity2 = self.body_2.GetPos_dt().y
          
        
        
    def updateActuator1(self, input1):
        #update actuator1
        self.control_force.SetMforce(input1)
      
    def updateActuator2(self, input2):
        #update actuator2
        pass

###############################################################################
        
class rotating_mass_spring():
     def __init__(self,body_size=0.3,spring_coef=50,rest_length=0,damping_coef=0.01,constant_rot = 0.0, prop_rot = 3.0, adjust = 0.1, rotor_length = 3):
         self.system = chrono.ChSystemNSC()
         self.title = ('rotating_mass_spring_system')
         
         # Set gravity
         self.system.Set_G_acc(chrono.ChVectorD(0, 0, 0))
         self.camara_vector = chrono.ChVectorD(-0.0, 4, 6)
         
         self.rotation_mode = 'proportional' #constant or proportional

         self.constant_factor = constant_rot
         self.proportional_factor = prop_rot

         self.table_color = chrono.ChColor(0.1 ,.3, .3)
         
         #Set parameters
         self.rotor_length = rotor_length
         self.rotor_width = .05
         self.body_1_size = body_size
         
         #create ground
         self.ground = chrono.ChBody()
         self.system.AddBody(self.ground)
         self.ground.SetIdentifier(-1)
         self.ground.SetBodyFixed(True)
         self.ground.SetCollide(False)
         
         #create table
         self.table = chrono.ChBodyEasyCylinder(1 ,self.rotor_length / 2, self.body_1_size / 2, 1,True,False)
         self.system.AddBody(self.table)
         self.table.SetBodyFixed(True)
         self.table.SetPos(chrono.ChVectorD(0, -self.body_1_size*2, 0))
         self.table.GetVisualShape(0).SetColor(self.table_color)
         
         #create pivot
         self.pivot = chrono.ChBodyEasyCylinder(1 ,self.body_1_size / 2, self.body_1_size, 1,True,False)
         self.system.AddBody(self.pivot)
         self.pivot.SetBodyFixed(True)
         self.pivot.SetPos(chrono.ChVectorD(0, -self.body_1_size*1.25, 0))
         self.pivot.GetVisualShape(0).SetColor(self.table_color)
         
         #create rotor
         self.rotor_1 = chrono.ChBodyEasyBox(self.rotor_length, self.rotor_width, self.rotor_width, 1, True, False)
         self.system.AddBody(self.rotor_1)

         #create visual objects (no physics)
         self.under_bar = chrono.ChBoxShape(self.rotor_length, self.rotor_width, self.rotor_width)
         self.rotor_1.AddVisualShape(self.under_bar, chrono.ChFrameD(chrono.ChVectorD(0, -self.body_1_size, 0)))

         self.side_bar_1 = chrono.ChBoxShape(self.rotor_width, self.body_1_size- self.rotor_width , self.rotor_width )
         self.rotor_1.AddVisualShape(self.side_bar_1, chrono.ChFrameD(chrono.ChVectorD(self.rotor_length / 2 - self.rotor_width / 2, -self.body_1_size / 2, 0)))

         self.side_bar_2 = chrono.ChBoxShape(self.rotor_width , self.body_1_size- self.rotor_width, self.rotor_width )
         self.rotor_1.AddVisualShape(self.side_bar_2, chrono.ChFrameD(chrono.ChVectorD(-self.rotor_length / 2 + self.rotor_width / 2, -self.body_1_size / 2, 0)))

         self.sph_1 = chrono.ChSphereShape(0.1)
         self.ground.AddVisualShape(self.sph_1, chrono.ChFrameD(chrono.ChVectorD(0, 0, 0)))

         # create body object
         self.body_1 = chrono.ChBodyEasyBox(self.body_1_size, self.body_1_size, self.body_1_size, rho_wood, True, False)
         self.system.AddBody(self.body_1)
         self.body_1_mass = self.body_1.GetMass()
         if self.proportional_factor>0:
           self.body_1_eq = m.sqrt(spring_coef/(self.body_1_mass*self.proportional_factor**2))
           self.body_1.SetPos(chrono.ChVectorD(self.body_1_eq+adjust, 0, 0))
         else:
           self.body_1.SetPos(chrono.ChVectorD(adjust, 0, 0))

         # Attach a visualization asset.
         self.box_1 = chrono.ChBoxShape(self.body_1_size , self.body_1_size , self.body_1_size )
         self.box_1.SetColor(chrono.ChColor(0.6, 0, 0))
         self.body_1.AddVisualShape(self.box_1)

         # Create a ChForce object
         self.control_force = chrono.ChForce()

         # Add the force to the body
         self.body_1.AddForce(self.control_force)


         # Create the spring between body_1 and ground. The spring end points are
         # specified in the body relative frames.
         self.spring_force = MySpringForce(spring_coef, rest_length, damping_coef)
         self.spring_1 = chrono.ChLinkTSDA()
         self.spring_1.Initialize(self.body_1, self.ground, True, chrono.ChVectorD(0, 0, 0), chrono.ChVectorD(0, 0, 0))
         self.spring_1.RegisterForceFunctor(self.spring_force)
         self.system.AddLink(self.spring_1)


         # Attach a visualization asset.
         self.spring_1_shape = chrono.ChSpringShape(0.05, 80, 15)
         self.spring_1_shape.SetColor(chrono.ChColor(0.3, 0, 0))
         self.spring_1.AddVisualShape(self.spring_1_shape)
         
         # Create prismatic link
         self.prismatic1 = chrono.ChLinkLockPrismatic()
         self.prismatic1.Initialize(self.rotor_1, self.body_1, chrono.ChCoordsysD(
             chrono.ChVectorD(0, 0, 0), chrono.Q_from_AngY(chrono.CH_C_PI_2)))
         self.system.AddLink(self.prismatic1)

         # Create the motor
         positionA1 = chrono.ChVectorD(0, 0, 0)
         axis = chrono.ChVectorD(1, 0, 0)
         self.rotmotor1 = chrono.ChLinkMotorRotationSpeed()

         # Connect the rotor and the stator and add the motor to the sys:
         self.rotmotor1.Initialize(self.ground,                # body A (slave)
                               self.rotor_1,               # body B (master)
                               chrono.ChFrameD(positionA1, chrono.Q_from_AngAxis(chrono.CH_C_PI / 2, axis)))  # motor frame, in abs. coords
         self.system.Add(self.rotmotor1)
         vw_constant = chrono.ChFunction_Const(0)
         self.rotmotor1.SetSpeedFunction(vw_constant)
         
         self.velocity = 0
         self.position = m.sqrt(self.body_1.GetPos().z**2+self.body_1.GetPos().x**2)
         
     def getState(self):
         #return the current state
         return np.array([self.position,self.velocity])
       
     def updateSystem(self, timestep = 1e-2):
         #Do step
         self.system.DoStepDynamics(timestep)
         
         #calculate state
         self.position = m.sqrt(self.body_1.GetPos().z**2+self.body_1.GetPos().x**2)
         self.velocity = (self.body_1.GetPos().z*self.body_1.GetPos_dt().z+self.body_1.GetPos().x*self.body_1.GetPos_dt().x)/self.position
         
         #adjust system
         if self.rotation_mode == 'proportional':
            vw_proportional = chrono.ChFunction_Const(self.proportional_factor*(self.position)+self.constant_factor)
            self.rotmotor1.SetSpeedFunction(vw_proportional)
         elif self.rotation_mode == 'constant':
            vw_constant = chrono.ChFunction_Const(self.constant_factor)
            self.rotmotor1.SetSpeedFunction(vw_constant)
       
     def updateActuator1(self, input1):
         #update actuator1
         self.control_force.SetMforce(input1)
       
     def updateActuator2(self, input2):
         #update actuator2
         pass       
    
###############################################################################

class balance_ball():
     def __init__(self,control_mode=3, start_pos = 1/32):
         self.system = chrono.ChSystemNSC()
         self.system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
         self.title = ('balance_ball_system')
         
         # Set gravity
         self.system.Set_G_acc(chrono.ChVectorD(0, -9.81, 0))
         self.camara_vector = chrono.ChVectorD(-0.8, 1.2, 1.5)

         # Set the simulation parameters
         self.beam_length = 1.0
         self.beam_thickness = 0.05
         self.ball_radius = 0.1
         self.friction_coefficient = 1.0

         self.control_mode = control_mode
         self.starting_pos_x = self.beam_length * start_pos

         # Create a material with the desired friction coefficient
         self.material = chrono.ChMaterialSurfaceNSC()
         self.material.SetFriction(self.friction_coefficient)

         # Create the ground body
         self.ground = chrono.ChBody()
         self.ground.SetBodyFixed(True)
         self.system.Add(self.ground)

         # Create the beam
         self.beam = chrono.ChBodyEasyBox(self.beam_thickness, self.beam_length, self.beam_thickness, rho_wood, True, True, self.material)
         self.beam.SetPos(chrono.ChVectorD(0, 0, 0))
         self.beam.SetRot(chrono.Q_from_AngAxis(chrono.CH_C_PI / 2, chrono.ChVectorD(0, 0, 1)))
         self.system.Add(self.beam)

         # Create the ball
         self.y_offset = self.ball_radius+self.beam_thickness / 2
         self.body_1 = chrono.ChBodyEasySphere(self.ball_radius, rho_wood, True, True, self.material)
         self.body_1.SetPos(chrono.ChVectorD(self.starting_pos_x, self.y_offset , 0))
         self.body_1.GetVisualShape(0).SetTexture(chrono.GetChronoDataFile("textures/bluewhite.png"))
         self.system.Add(self.body_1)

         # Create a revolute joint between the beam and the ground
         self.revolute = chrono.ChLinkLockRevolute()
         self.revolute.Initialize(self.beam, self.ground, chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), chrono.Q_from_AngAxis(chrono.CH_C_PI / 2, chrono.ChVectorD(0, 0, 1))))
         self.system.AddLink(self.revolute)

         # Create the motor
         positionA1 = chrono.ChVectorD(0, 0, 0) 
         if self.control_mode == 1:
           self.rotmotor1 = chrono.ChLinkMotorRotationAngle()
         elif self.control_mode == 2:
           self.rotmotor1 = chrono.ChLinkMotorRotationSpeed()
         else:
           self.rotmotor1 = chrono.ChLinkMotorRotationTorque()

         # Connect the rotor and the stator and add the motor to the sys:
         self.rotmotor1.Initialize(self.ground,                # body A (slave)
                               self.beam,               # body B (master)
                               chrono.ChFrameD(positionA1)  # motor frame, in abs. coords
         )
         self.system.Add(self.rotmotor1)
         
     def getState(self):
         #return the current state
         try: 
           return np.array([self.position,self.velocity,self.beam_angle,self.beam_angle_vel])
         except:
           return np.array([0,0,0,0])
       
     def updateSystem(self, timestep = 1e-2):
         #Do step
         self.system.DoStepDynamics(timestep)
         
         #fetch current angle
         self.angle_x = shiftAngle(self.revolute.GetRelAngle())
         
         #calculate state
         self.position = self.body_1.GetPos().x/m.cos(self.angle_x)
         self.velocity = self.body_1.GetPos_dt().x/m.cos(self.angle_x)
         self.beam_angle = self.angle_x
         self.beam_angle_vel = self.revolute.GetRelWvel().z
       
     def updateActuator1(self, input1):
         #update actuator1
         f_input1 = chrono.ChFunction_Const(input1)
         if self.control_mode == 1:
           self.rotmotor1.SetAngleFunction(f_input1)
         elif self.control_mode == 2:
           self.rotmotor1.SetSpeedFunction(f_input1)
         else:
           self.rotmotor1.SetMotorFunction(f_input1)
           
     def updateActuator2(self, input2):
         #update actuator2
         pass       
       
###############################################################################

class cart_inverted_pendulum():
     def __init__(self,second_pendulum = False, pendulum1_length = 0.6, pendulum2_length = 0.6, high_kick = 2.0):
         self.system = chrono.ChSystemNSC()
         self.system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
         self.title = ('cart_inverted_pendulum')
         
         # Set gravity
         self.system.Set_G_acc(chrono.ChVectorD(0, -9.81, 0))
         self.camara_vector = chrono.ChVectorD(-0.8, 1.2, 1.5)

         self.second_pendulum = second_pendulum
         self.pendulum1_angle = 0.0
         self.pendulum2_angle = 0.0

         # Set up ground
         self.ground = chrono.ChBodyEasyBox(3, 0.2, 1, rho_wood, True, False)
         self.ground.SetPos(chrono.ChVectorD(0, -0.1, 0))
         self.ground.SetBodyFixed(True)
         linebox = chrono.ChBoxShape(0.01, 0.205, 1.005)
         linebox.SetColor(chrono.ChColor(0.6, 0, 0))
         self.ground.AddVisualShape(linebox, chrono.ChFrameD(chrono.ChVectorD(0, 0, 0), chrono.QUNIT))
         self.system.Add(self.ground)

         # Set up cart
         self.cart = chrono.ChBodyEasyBox(0.2, 0.2, 0.4, rho_wood, True, False)
         self.cart.SetPos(chrono.ChVectorD(0, 0.1, 0))
         self.cart.SetPos_dt(chrono.ChVectorD(high_kick, 0, 0))
         linebox2 = chrono.ChBoxShape(0.01, 0.205, 0.405)
         linebox2.SetColor(chrono.ChColor(0.6, 0, 0))
         self.cart.AddVisualShape(linebox2, chrono.ChFrameD(chrono.ChVectorD(0, 0, 0), chrono.QUNIT))
         self.system.Add(self.cart)
        
         # Create a ChForce object
         self.control_force = chrono.ChForce()

         # Add the force to the body
         self.cart.AddForce(self.control_force)

         # Set up first pendulum
         self.pendulum1 = chrono.ChBodyEasyBox(0.02, pendulum1_length, 0.02, rho_wood, True, False)
         self.pendulum1.SetPos(chrono.ChVectorD(0, pendulum1_length / 2 + 0.2, 0))
         self.pendulum1.SetRot(chrono.Q_from_AngAxis(self.pendulum1_angle, chrono.ChVectorD(0, 0, 1)))
         self.system.Add(self.pendulum1)

         # Set up revolute joint between cart and first pendulum
         self.revolute1 = chrono.ChLinkLockRevolute()
         self.revolute1.Initialize(self.cart, self.pendulum1, chrono.ChCoordsysD(chrono.ChVectorD(0, 0.2, 0), chrono.Q_from_AngZ(chrono.CH_C_PI_2)))
         self.system.Add(self.revolute1)

         if self.second_pendulum:
           # Set up second pendulum
           self.pendulum2 = chrono.ChBodyEasyBox(0.02, pendulum2_length, 0.02, rho_wood, True, False)
           self.pendulum2.SetPos(chrono.ChVectorD(0, pendulum2_length / 2 + pendulum1_length + 0.2, 0))
           self.pendulum2.SetRot(chrono.Q_from_AngAxis(self.pendulum2_angle, chrono.ChVectorD(0, 0, 1)))
           self.system.Add(self.pendulum2)
           
           # Set up revolute joint between first and second pendulum
           self.revolute2 = chrono.ChLinkLockRevolute()
           self.revolute2.Initialize(self.pendulum1, self.pendulum2, chrono.ChCoordsysD(chrono.ChVectorD(0, pendulum1_length + 0.2, 0), chrono.Q_from_AngZ(chrono.CH_C_PI_2)))
           self.system.Add(self.revolute2)

         # Create prismatic joints between ground and cart
         self.prismatic1 = chrono.ChLinkLockPrismatic()
         self.prismatic1.Initialize(self.ground, self.cart, chrono.ChCoordsysD(
             chrono.ChVectorD(0, 0, 0), chrono.Q_from_AngY(chrono.CH_C_PI_2)))
         self.system.AddLink(self.prismatic1)
         
         # initiate state variables
         self.cart_position = 0. 
         self.cart_velocity = 0.
         
         self.pend1_angle = 0.
         self.pend1_angle_vel = 0. 
         
         if self.second_pendulum:
           self.pend2_angle = 0.
           self.pend2_angle_vel = 0. 
           
     def getState(self):
         #return the current state
         if self.second_pendulum:
           state = np.array([ self.cart_position,
                              self.cart_velocity,
                              self.pend1_angle,
                              self.pend1_angle_vel,
                              self.pend2_angle,
                              self.pend2_angle_vel])
         else:
           state = np.array([ self.cart_position,
                              self.cart_velocity,
                              self.pend1_angle,
                              self.pend1_angle_vel])
         return state
       
     def updateSystem(self, timestep = 1e-2):
         #Do step
         self.system.DoStepDynamics(timestep)
         
         #calculate state
         self.cart_position = self.cart.GetPos().x
         self.cart_velocity = self.cart.GetPos_dt().x
         self.pend1_angle = shiftAngle(self.revolute1.GetRelAngle())
         self.pend1_angle_vel = self.revolute1.GetRelWvel().z
         if self.second_pendulum:
           self.pend2_angle = shiftAngle(self.revolute2.GetRelAngle())
           self.pend2_angle_vel = self.revolute2.GetRelWvel().z
         
     def updateActuator1(self, input1):
         #update actuator1
         F_input = chrono.ChFunction_Const(input1)
         self.control_force.SetF_x(F_input)
           
     def updateActuator2(self, input2):
         #update actuator2
         pass      
            
###############################################################################

class balance_plate():
     def __init__(self, speed = 10, width = 0.1):
         self.system = chrono.ChSystemNSC()
         self.system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
         self.title = ('balance_plate_system')
         
         # Set gravity
         self.system.Set_G_acc(chrono.ChVectorD(0, -9.81, 0))
         self.camara_vector = chrono.ChVectorD(-0.8, 1.2, 1.5)

         # Set the simulation parameters
         self.beam_length = 2.0
         self.beam_thickness = 0.1
         self.ball_radius = 0.1
         self.friction_coefficient = 1.0
         self.starting_pos_x = 0.0

         # Create a material with the desired friction coefficient
         self.material = chrono.ChMaterialSurfaceNSC()
         self.material.SetFriction(self.friction_coefficient)

         # Create the ground body
         self.ground = chrono.ChBody()
         self.ground.SetBodyFixed(True)
         self.system.Add(self.ground)
         
         # Create the connect body
         self.connect = chrono.ChBody()
         self.system.Add(self.connect)

         # Create the beam
         self.beam = chrono.ChBodyEasyBox(self.beam_thickness / 2, self.beam_length / 2, self.beam_length  / 2, rho_wood, True, True, self.material)
         self.beam.SetPos(chrono.ChVectorD(0, 0, 0))
         self.beam.SetRot(chrono.Q_from_AngAxis(chrono.CH_C_PI / 2, chrono.ChVectorD(0, 0, 1)))
         self.system.Add(self.beam)

         # Create the ball
         self.y_offset = self.ball_radius+self.beam_thickness / 2
         self.body_1 = chrono.ChBodyEasySphere(self.ball_radius, rho_wood, True, True, self.material)
         self.body_1.SetPos(chrono.ChVectorD(self.starting_pos_x, self.y_offset , 0))
         self.body_1.GetVisualShape(0).SetTexture(chrono.GetChronoDataFile("textures/bluewhite.png"))
         self.system.Add(self.body_1)

         # Create a revolute joint between the beam and the connect
         self.revolute1 = chrono.ChLinkLockRevolute()
         self.revolute1.Initialize(self.beam, self.connect, chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0))) #, chrono.Q_from_AngAxis(chrono.CH_C_PI / 2, chrono.ChVectorD(0, 0, 0))
         self.system.AddLink(self.revolute1)

         positionA1 = chrono.ChVectorD(0, 0, 0) 
         self.rotmotor1 = chrono.ChLinkMotorRotationSpeed()

         # Connect the rotor and the stator and add the motor to the sys:
         self.rotmotor1.Initialize(self.connect,                # body A (slave)
                               self.beam,               # body B (master)
                               chrono.ChFrameD(positionA1)  # motor frame, in abs. coords
         )
         self.system.Add(self.rotmotor1)
         
         # Create a revolute joint between the connect and the ground
         self.revolute2 = chrono.ChLinkLockRevolute()
         self.revolute2.Initialize(self.connect, self.ground, chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), chrono.Q_ROTATE_Z_TO_X))
         self.system.AddLink(self.revolute2)
         
         self.rotmotor2 = chrono.ChLinkMotorRotationSpeed()

         # Connect the rotor and the stator and add the motor to the sys:
         self.rotmotor2.Initialize(self.ground,                # body A (slave)
                               self.connect,               # body B (master)
                               chrono.ChFrameD(positionA1,chrono.Q_ROTATE_Z_TO_X)  # motor frame, in abs. coords
         )
         self.system.Add(self.rotmotor2)
         
         self.angle_x = 0
         self.angle_z = 0
         
         self.width = width 
         self.speed = speed
         
     def getState(self):
         #return the current state
         try: 
           return np.array([self.position_x,self.velocity_x,self.position_z,self.velocity_z])
         except:
           return np.array([0,0,0,0])
       
     def updateSystem(self, timestep = 1e-2):
         #Do step
         self.system.DoStepDynamics(timestep)
         
         #fetch current angle
         self.angle_x = shiftAngle(self.revolute1.GetRelAngle())
         self.angle_z = shiftAngle(self.revolute2.GetRelAngle())
         
         #calculate state
         self.position_x = self.body_1.GetPos().x/m.cos(self.angle_x)
         self.velocity_x = self.body_1.GetPos_dt().x/m.cos(self.angle_x)
         self.position_z = self.body_1.GetPos().z/m.cos(self.angle_z)
         self.velocity_z = self.body_1.GetPos_dt().z/m.cos(self.angle_z)
         
         ### note ###
         #
         # These positions and velocities are only approximate projection.
         # The angles affect eachother, which creates a non linear interaction.
         # These projects are accurate around 0.
         
     def updateActuator1(self, input1):
         #update actuator1
         d_theta = input1 - self.angle_x
         f_input1 = chrono.ChFunction_Const(-self.speed*m.tanh(d_theta/self.width))
         self.rotmotor1.SetSpeedFunction(f_input1)
         
     def updateActuator2(self, input2):
         #update actuator2
         d_theta = input2 - self.angle_z
         f_input2 = chrono.ChFunction_Const(-self.speed*m.tanh(d_theta/self.width))
         self.rotmotor2.SetSpeedFunction(f_input2)     

###############################################################################

class rotation_inverted_pendulum():
     def __init__(self,second_pendulum = False, pendulum1_length = 0.6, pendulum2_length = 0.6, high_kick = 300.0):
         self.system = chrono.ChSystemNSC()
         self.system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
         self.title = ('rotation_inverted_pendulum')
         
         # Set gravity
         self.system.Set_G_acc(chrono.ChVectorD(0, -9.81, 0))
         self.camara_vector = chrono.ChVectorD(-0.8, 1.2, 1.5)

         

         self.second_pendulum = second_pendulum
         self.pendulum1_angle = 0.0
         self.pendulum2_angle = 0.0
         
         self.arm_length = 0.3

          
         # Set up ground
         self.ground = chrono.ChBodyEasyCylinder(1 ,self.arm_length *.8, 0.1, rho_wood,True,False)
         self.ground.SetPos(chrono.ChVectorD(0, -0.1, 0))
         self.ground.SetBodyFixed(True)
         linebox = chrono.ChBoxShape(2*self.arm_length *0.8+0.01, 0.105, .01)
         linebox.SetColor(chrono.ChColor(0.6, 0, 0))
         self.ground.AddVisualShape(linebox, chrono.ChFrameD(chrono.ChVectorD(0, 0, 0), chrono.QUNIT))
         self.system.Add(self.ground)
         self.table_color = chrono.ChColor(0.1 ,.3, .3)
         self.ground.GetVisualShape(0).SetColor(self.table_color)
         
         # Set up pivot
         self.pivot = chrono.ChBodyEasyCylinder(1 , self.arm_length *.2, .2, rho_wood,True,False)
         self.system.AddBody(self.pivot)
         self.pivot.SetBodyFixed(True)
         self.pivot.SetPos(chrono.ChVectorD(0, 0, 0))
         self.pivot.GetVisualShape(0).SetColor(self.table_color)

         # Set up arm
         self.arm = chrono.ChBodyEasyBox(self.arm_length, 0.03, 0.03, rho_wood, True, False)
         self.arm.SetPos(chrono.ChVectorD(self.arm_length/2, 0.1, 0))
         self.arm.SetRot_dt(chrono.Q_from_AngY(chrono.CH_C_PI_2)*high_kick)
         self.system.Add(self.arm)        

         # Set up first pendulum
         self.pendulum1 = chrono.ChBodyEasyBox(0.02, pendulum1_length, 0.02, rho_wood, True, False)
         self.pendulum1.SetPos(chrono.ChVectorD(self.arm_length, pendulum1_length / 2 + 0.115, 0))
         self.pendulum1.SetRot(chrono.Q_from_AngAxis(self.pendulum1_angle, chrono.ChVectorD(0, 0, 1)))
         self.system.Add(self.pendulum1)

         # Set up revolute joint between arm and first pendulum
         self.revolute1 = chrono.ChLinkLockRevolute()
         self.revolute1.Initialize(self.arm, self.pendulum1, chrono.ChCoordsysD(chrono.ChVectorD(self.arm_length, 0.115, 0), chrono.Q_from_AngY(chrono.CH_C_PI_2)))
         self.system.Add(self.revolute1)


         if self.second_pendulum:
           # Set up second pendulum
           self.pendulum2 = chrono.ChBodyEasyBox(0.02, pendulum2_length, 0.02, rho_wood, True, False)
           self.pendulum2.SetPos(chrono.ChVectorD(self.arm_length, pendulum2_length / 2 + pendulum1_length + 0.115, 0))
           self.pendulum2.SetRot(chrono.Q_from_AngAxis(self.pendulum2_angle, chrono.ChVectorD(0, 0, 1)))
           self.system.Add(self.pendulum2)
           
           # Set up revolute joint between first and second pendulum
           self.revolute2 = chrono.ChLinkLockRevolute()
           self.revolute2.Initialize(self.pendulum1, self.pendulum2, chrono.ChCoordsysD(chrono.ChVectorD(self.arm_length, pendulum1_length + 0.115, 0), chrono.Q_from_AngY(chrono.CH_C_PI_2)))
           self.system.Add(self.revolute2)

         # Create a revolute joint between the arm and the ground
         self.revolute = chrono.ChLinkLockRevolute()
         self.revolute.Initialize(self.arm, self.ground, chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0),chrono.Q_ROTATE_Z_TO_Y))
         self.system.AddLink(self.revolute)

         # Create the motor
         self.rotmotor1 = chrono.ChLinkMotorRotationTorque()
         positionA1 = chrono.ChVectorD(0, 0, 0) 
           
         # Connect the rotor and the stator and add the motor to the sys:
         self.rotmotor1.Initialize(self.ground,                # body A (slave)
                               self.arm,               # body B (master)
                               chrono.ChFrameD(positionA1,chrono.Q_ROTATE_Z_TO_Y)  # motor frame, in abs. coords,
         )
         self.system.Add(self.rotmotor1)
         
         # initiate state variables
         self.arm_angle = 0. 
         self.arm_angle_vel = 0.
         
         self.pend1_angle = 0.
         self.pend1_angle_vel = 0. 
         
         if self.second_pendulum:
           self.pend2_angle = 0.
           self.pend2_angle_vel = 0. 
           
     def getState(self):
         #return the current state
         if self.second_pendulum:
           state = np.array([ self.arm_angle,
                              self.arm_angle_vel,
                              self.pend1_angle,
                              self.pend1_angle_vel,
                              self.pend2_angle,
                              self.pend2_angle_vel])
         else:
           state = np.array([ self.arm_angle,
                              self.arm_angle_vel,
                              self.pend1_angle,
                              self.pend1_angle_vel])
         return state
       
     def updateSystem(self, timestep = 1e-2):
         #Do step
         self.system.DoStepDynamics(timestep)
         
         #calculate state
         self.arm_angle = shiftAngle(self.revolute.GetRelAngle())
         self.arm_angle_vel = self.revolute.GetRelWvel().z
         
         self.pend1_angle = shiftAngle(self.revolute1.GetRelAngle())
         self.pend1_angle_vel = self.revolute1.GetRelWvel().z
         
         if self.second_pendulum:
           self.pend2_angle = shiftAngle(self.revolute2.GetRelAngle())
           self.pend2_angle_vel = self.revolute2.GetRelWvel().z
         
     def updateActuator1(self, input1):
         #update actuator1
         f_input1 = chrono.ChFunction_Const(input1)
         self.rotmotor1.SetMotorFunction(f_input1)
         pass
           
     def updateActuator2(self, input2):
         #update actuator2
         pass      
             
###############################################################################

class flywheel_inverted_pendulum():
      def __init__(self,second_pendulum = False, pendulum1_length = 0.6, pendulum2_length = 0.6, high_kick = 100.0):
          self.system = chrono.ChSystemNSC()
          self.system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
          self.title = ('flywheel_inverted_pendulum')
          # Set gravity
          self.system.Set_G_acc(chrono.ChVectorD(0, -9.81, 0))
          self.camara_vector = chrono.ChVectorD(-0.8, 1.2, 1.5)

          self.second_pendulum = second_pendulum
          self.pendulum1_angle = 0.0
          self.pendulum2_angle = 0.0
          self.flywheel_radius = 0.1
          self.flywheel_thickness = 0.01 

          # Set up ground
          self.ground = chrono.ChBodyEasyBox(3, 0.2, 1, rho_wood, True, False)
          self.ground.SetPos(chrono.ChVectorD(0, -0.1, 0))
          self.ground.SetBodyFixed(True)
          self.system.Add(self.ground)

          # Set up first pendulum
          self.pendulum1 = chrono.ChBodyEasyBox(0.02, pendulum1_length, 0.02, rho_wood, True, False)
          self.pendulum1.SetPos(chrono.ChVectorD(0, pendulum1_length / 2 + 0.0, 0))
          self.pendulum1.SetRot(chrono.Q_from_AngAxis(self.pendulum1_angle, chrono.ChVectorD(0, 0, 1)))
          self.pendulum1.SetRot_dt(chrono.Q_from_AngZ(chrono.CH_C_PI_2)*high_kick)
          self.system.Add(self.pendulum1)

          # Set up revolute joint between ground and first pendulum
          self.revolute1 = chrono.ChLinkLockRevolute()
          self.revolute1.Initialize(self.ground, self.pendulum1, chrono.ChCoordsysD(chrono.ChVectorD(0, 0.0, 0), chrono.Q_from_AngZ(chrono.CH_C_PI_2)))
          self.system.Add(self.revolute1)
          
          # Set up flywheel
          self.fly = chrono.ChBodyEasyCylinder(1 , self.flywheel_radius, self.flywheel_thickness, rho_iron,True,False)
          linebox = chrono.ChBoxShape(0.01, self.flywheel_thickness+0.015, self.flywheel_radius)
          linebox.SetColor(chrono.ChColor(0.6, 0, 0))
          self.fly.AddVisualShape(linebox, chrono.ChFrameD(chrono.ChVectorD(0,0,-self.flywheel_radius /2), chrono.QUNIT))
          self.system.AddBody(self.fly)
          self.fly.SetPos(chrono.ChVectorD(0, pendulum1_length + 0.0, 0))
          self.fly.SetRot(chrono.Q_from_AngX(chrono.CH_C_PI_2))
          self.fly_color = chrono.ChColor(0.6 ,.6, .9)
          self.fly.GetVisualShape(0).SetColor(self.fly_color)
          
          # Set up revolute joint between flywheel and first pendulum
          self.revoluteF = chrono.ChLinkLockRevolute()
          self.revoluteF.Initialize(self.pendulum1, self.fly, chrono.ChCoordsysD(chrono.ChVectorD(0, pendulum1_length + 0.0, 0), chrono.Q_from_AngZ(chrono.CH_C_PI_2)))
          self.system.Add(self.revoluteF)
          
          # Create the motor
          self.rotmotor1 = chrono.ChLinkMotorRotationTorque()
          positionA1 = chrono.ChVectorD(0, pendulum1_length + 0.0, 0)
            
          # Connect the rotor and the stator and add the motor to the sys:
          self.rotmotor1.Initialize(self.pendulum1,             # body A (slave)
                                    self.fly,          # body B (master)
                                    chrono.ChFrameD(positionA1)  # motor frame, in abs. coords,
          )
          self.system.Add(self.rotmotor1)

          if self.second_pendulum:
            # Set up second pendulum
            self.pendulum2 = chrono.ChBodyEasyBox(0.02, pendulum2_length, 0.02, rho_wood, True, False)
            self.pendulum2.SetPos(chrono.ChVectorD(0, pendulum2_length / 2 + pendulum1_length + 0.0, 0))
            self.pendulum2.SetRot(chrono.Q_from_AngAxis(self.pendulum2_angle, chrono.ChVectorD(0, 0, 1)))
            self.system.Add(self.pendulum2)
            
            # Set up revolute joint between first and second pendulum
            self.revolute2 = chrono.ChLinkLockRevolute()
            self.revolute2.Initialize(self.pendulum1, self.pendulum2, chrono.ChCoordsysD(chrono.ChVectorD(0, pendulum1_length, 0), chrono.Q_from_AngZ(chrono.CH_C_PI_2)))
            self.system.Add(self.revolute2)

          # initiate state variables
          self.fly_angle = 0. 
          self.fly_angle_vel = 0.
          
          self.pend1_angle = 0.
          self.pend1_angle_vel = 0. 
          
          if self.second_pendulum:
            self.pend2_angle = 0.
            self.pend2_angle_vel = 0. 
          
      def getState(self):
          #return the current state
          if self.second_pendulum:
            state = np.array([ self.fly_angle,
                               self.fly_angle_vel,
                               self.pend1_angle,
                               self.pend1_angle_vel,
                               self.pend2_angle,
                               self.pend2_angle_vel])
          else:
            state = np.array([ self.fly_angle,
                               self.fly_angle_vel,
                               self.pend1_angle,
                               self.pend1_angle_vel])
          return state
        
      def updateSystem(self, timestep = 1e-2):
          #Do step
          self.system.DoStepDynamics(timestep)
          
          #calculate state
          self.fly_angle = shiftAngle(self.revoluteF.GetRelAngle())
          self.fly_angle_vel = self.revoluteF.GetRelWvel().z
          self.pend1_angle = shiftAngle(self.revolute1.GetRelAngle())
          self.pend1_angle_vel = self.revolute1.GetRelWvel().z
          if self.second_pendulum:
            self.pend2_angle = shiftAngle(self.revolute2.GetRelAngle())
            self.pend2_angle_vel = self.revolute2.GetRelWvel().z
          
      def updateActuator1(self, input1):
          #update actuator1
          f_input1 = chrono.ChFunction_Const(input1)
          self.rotmotor1.SetMotorFunction(f_input1)
          pass
            
      def updateActuator2(self, input2):
          #update actuator2
          pass      
                
###############################################################################

class stacked_inverted_pendulum():
      def __init__(self,num_pendulum = 1, pendulum_length = 0.6, high_kick = 2.0):
          self.system = chrono.ChSystemNSC()
          self.system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
          self.title = ('stacked_inverted_pendulum')
          
          # Set gravity
          self.system.Set_G_acc(chrono.ChVectorD(0, -9.81, 0))
          self.camara_vector = chrono.ChVectorD(-0.8, 1.2, 1.5)

          self.second_pendulum = (num_pendulum>1)
          self.third_pendulum = (num_pendulum>2)
          self.fourth_pendulum = (num_pendulum>3)
          
          if self.fourth_pendulum:
            self.camara_vector = chrono.ChVectorD(-0.8, 1.5, 4)
          elif self.third_pendulum: 
            self.camara_vector = chrono.ChVectorD(-0.8, 1.2, 3)
          
          self.pendulum_angle = 0.0

          # Set up ground
          self.ground = chrono.ChBodyEasyBox(0.5, 0.2, 0.5, rho_wood, True, False)
          self.ground.SetPos(chrono.ChVectorD(0, -0.1, -.25-0.01))
          self.ground.SetBodyFixed(True)
          self.system.Add(self.ground)
          self.table_color = chrono.ChColor(0.3 ,.6, .6)
          self.ground.GetVisualShape(0).SetColor(self.table_color)
          
          sph_1 = chrono.ChSphereShape(0.02)
          self.ground.AddVisualShape(sph_1, chrono.ChFrameD(chrono.ChVectorD(0, 0.1, 0.25+0.01)))

          # Set up first pendulum
          self.pendulum1 = chrono.ChBodyEasyBox(0.02, pendulum_length, 0.02, rho_wood, True, False)
          self.pendulum1.SetPos(chrono.ChVectorD(0, pendulum_length / 2 + 0.0, 0))
          self.pendulum1.SetRot(chrono.Q_from_AngAxis(self.pendulum_angle, chrono.ChVectorD(0, 0, 1)))
          self.pendulum1.SetRot_dt(chrono.Q_from_AngZ(chrono.CH_C_PI_2)*high_kick)
          self.system.Add(self.pendulum1)

          # Set up revolute joint between ground and first pendulum
          self.revolute1 = chrono.ChLinkLockRevolute()
          self.revolute1.Initialize(self.ground, self.pendulum1, chrono.ChCoordsysD(chrono.ChVectorD(0, 0.0, 0), chrono.Q_from_AngZ(chrono.CH_C_PI_2)))
          self.system.Add(self.revolute1)
          
          # Create the motor
          self.rotmotor1 = chrono.ChLinkMotorRotationTorque()
          positionA1 = chrono.ChVectorD(0, 0, 0)
            
          # Connect the rotor and the stator and add the motor to the sys:
          self.rotmotor1.Initialize(self.ground,                # body A (slave)
                                self.pendulum1,               # body B (master)
                                chrono.ChFrameD(positionA1)  # motor frame, in abs. coords,
          )
          self.system.Add(self.rotmotor1)

          if self.second_pendulum:
            # Set up second pendulum
            self.pendulum2 = chrono.ChBodyEasyBox(0.02, pendulum_length, 0.02, rho_wood, True, False)
            self.pendulum2.SetPos(chrono.ChVectorD(0, 3* pendulum_length / 2, 0))
            self.pendulum2.SetRot(chrono.Q_from_AngAxis(self.pendulum_angle, chrono.ChVectorD(0, 0, 1)))
            self.system.Add(self.pendulum2)
            
            # Set up revolute joint between first and second pendulum
            self.revolute2 = chrono.ChLinkLockRevolute()
            self.revolute2.Initialize(self.pendulum1, self.pendulum2, chrono.ChCoordsysD(chrono.ChVectorD(0, pendulum_length, 0), chrono.Q_from_AngZ(chrono.CH_C_PI_2)))
            self.system.Add(self.revolute2)
            
            if self.third_pendulum:
              # Set up third pendulum
              self.pendulum3 = chrono.ChBodyEasyBox(0.02, pendulum_length, 0.02, rho_wood, True, False)
              self.pendulum3.SetPos(chrono.ChVectorD(0, 5* pendulum_length / 2, 0))
              self.pendulum3.SetRot(chrono.Q_from_AngAxis(self.pendulum_angle, chrono.ChVectorD(0, 0, 1)))
              self.system.Add(self.pendulum3)
              
              # Set up revolute joint between second and third pendulum
              self.revolute3 = chrono.ChLinkLockRevolute()
              self.revolute3.Initialize(self.pendulum2, self.pendulum3, chrono.ChCoordsysD(chrono.ChVectorD(0, 2*pendulum_length, 0), chrono.Q_from_AngZ(chrono.CH_C_PI_2)))
              self.system.Add(self.revolute3)
              
              if self.fourth_pendulum:
                # Set up fourth pendulum
                self.pendulum4 = chrono.ChBodyEasyBox(0.02, pendulum_length, 0.02, rho_wood, True, False)
                self.pendulum4.SetPos(chrono.ChVectorD(0, 7* pendulum_length / 2, 0))
                self.pendulum4.SetRot(chrono.Q_from_AngAxis(self.pendulum_angle, chrono.ChVectorD(0, 0, 1)))
                self.system.Add(self.pendulum4)
                
                # Set up revolute joint between third and fourth pendulum
                self.revolute4 = chrono.ChLinkLockRevolute()
                self.revolute4.Initialize(self.pendulum3, self.pendulum4, chrono.ChCoordsysD(chrono.ChVectorD(0, 3*pendulum_length, 0), chrono.Q_from_AngZ(chrono.CH_C_PI_2)))
                self.system.Add(self.revolute4)
          
          self.pend1_angle = 0.
          self.pend1_angle_vel = 0. 
          
          if self.second_pendulum:
            self.pend2_angle = 0.
            self.pend2_angle_vel = 0. 
            
            if self.third_pendulum:
              self.pend3_angle = 0.
              self.pend3_angle_vel = 0.
              
              if self.fourth_pendulum:
                self.pend4_angle = 0.
                self.pend4_angle_vel = 0.        
          
      def getState(self):
          #return the current state
          if self.fourth_pendulum:
            state = np.array([ self.pend1_angle,
                               self.pend1_angle_vel,
                               self.pend2_angle,
                               self.pend2_angle_vel,
                               self.pend3_angle,
                               self.pend3_angle_vel,
                               self.pend4_angle,
                               self.pend4_angle_vel])
          
          elif self.third_pendulum:
            state = np.array([ self.pend1_angle,
                               self.pend1_angle_vel,
                               self.pend2_angle,
                               self.pend2_angle_vel,
                               self.pend3_angle,
                               self.pend3_angle_vel])
          
          elif self.second_pendulum:
            state = np.array([ self.pend1_angle,
                               self.pend1_angle_vel,
                               self.pend2_angle,
                               self.pend2_angle_vel])
          else:
            state = np.array([ self.pend1_angle,
                               self.pend1_angle_vel])
          return state
        
      def updateSystem(self, timestep = 1e-2):
          #Do step
          self.system.DoStepDynamics(timestep)
          
          #calculate state
          self.pend1_angle = shiftAngle(self.revolute1.GetRelAngle())
          self.pend1_angle_vel = self.revolute1.GetRelWvel().z
          if self.second_pendulum:
            self.pend2_angle = shiftAngle(self.revolute2.GetRelAngle())
            self.pend2_angle_vel = self.revolute2.GetRelWvel().z
            if self.third_pendulum:
              self.pend3_angle = shiftAngle(self.revolute3.GetRelAngle())
              self.pend3_angle_vel = self.revolute3.GetRelWvel().z
              if self.fourth_pendulum:
                self.pend4_angle = shiftAngle(self.revolute4.GetRelAngle())
                self.pend4_angle_vel = self.revolute4.GetRelWvel().z
          
      def updateActuator1(self, input1):
          #update actuator1
          f_input1 = chrono.ChFunction_Const(input1)
          self.rotmotor1.SetMotorFunction(f_input1)
          pass
            
      def updateActuator2(self, input2):
          #update actuator2
          pass      
                
###############################################################################

class cylinder_stability():
     def __init__(self, length = 2, width = 0.3, radius = 0.4):
         self.system = chrono.ChSystemNSC()
         self.system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
         self.title = ('cylinder_stability')
         
         # Set gravity
         self.system.Set_G_acc(chrono.ChVectorD(0, -9.81, 0))
         self.camara_vector = chrono.ChVectorD(2+length, 0.9, 1.2)

         # Set the simulation parameters
         self.beam_length = length
         self.beam_thickness = width
         self.cyl_radius = radius
         self.cyl_height = 2*width
         self.friction_coefficient = 1.0
         self.table_color = chrono.ChColor(0.1 ,.1, .1)

         # Create a material with the desired friction coefficient
         self.material = chrono.ChMaterialSurfaceNSC()
         self.material.SetFriction(self.friction_coefficient)

         # Create the beam
         self.beam = chrono.ChBodyEasyBox(self.beam_length,self.beam_thickness,  self.beam_thickness , rho_wood, True, True, self.material)
         self.beam.SetPos(chrono.ChVectorD(0, self.cyl_radius+self.beam_length/2, 0.001))
         self.beam.SetRot(chrono.Q_from_AngAxis(chrono.CH_C_PI / 2, chrono.ChVectorD(0, 0, 1)))
         self.system.Add(self.beam)

         # Create the cylinder
         self.cyl = chrono.ChBodyEasyCylinder(0, self.cyl_radius, self.cyl_height, rho_wood,True,True,self.material)
         self.cyl.SetPos(chrono.ChVectorD(0, 0, 0))
         self.cyl.GetVisualShape(0).SetTexture(chrono.GetChronoDataFile("textures/bluewhite.png"))
         self.cyl.SetBodyFixed(True)
         self.system.Add(self.cyl)
         
     def getState(self):
         #return the current state
         try: 
           return np.array([self.beam_angle,self.beam_angle_vel])
         except:
           return np.array([0,0])
       
     def updateSystem(self, timestep = 1e-2):
         #Do step
         self.system.DoStepDynamics(timestep)

         #calculate state
         self.beam_angle = chrono.Q_to_Euler123(self.beam.GetRot()).y 
         self.beam_angle_vel = chrono.Q_to_Euler123(self.beam.GetRot_dt()).y
       
     def updateActuator1(self, input1):
         #update actuator1
         pass
           
     def updateActuator2(self, input2):
         #update actuator2
         pass       

###############################################################################

class balance_slide():
     def __init__(self, body_size = 0.2, start_pos = 1/32):
         self.system = chrono.ChSystemNSC()
         self.system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
         self.title = ('balance_slide_system')
         
         # Set gravity
         self.system.Set_G_acc(chrono.ChVectorD(0, -9.81, 0))
         self.camara_vector = chrono.ChVectorD(-0.8, 1.2, 1.5)

         # Set the simulation parameters
         self.beam_length = 2.0
         self.beam_thickness = 0.05

         self.starting_pos_x = self.beam_length * start_pos
         
         # Set parameters
         self.body_1_size = body_size
         
         # Create the beam
         self.beam = chrono.ChBodyEasyBox(self.beam_thickness, self.beam_length, self.beam_thickness, rho_wood, True, False)
         self.beam.SetPos(chrono.ChVectorD(0, 0, 0))
         self.beam.SetRot(chrono.Q_from_AngAxis(chrono.CH_C_PI / 2, chrono.ChVectorD(0, 0, 1)))
         self.system.Add(self.beam)

         # Create the box
         self.body_1 = chrono.ChBodyEasyBox(self.body_1_size, self.body_1_size, self.body_1_size, rho_wood, True, False)
         self.body_1.SetPos(chrono.ChVectorD(self.starting_pos_x, 0 , 0))
         self.body_1.GetVisualShape(0).SetTexture(chrono.GetChronoDataFile("textures/bluewhite.png"))
         self.system.Add(self.body_1)

         # Create the ground body
         self.ground = chrono.ChBody()
         self.ground.SetBodyFixed(True)
         self.system.Add(self.ground)
         
         # Create a revolute joint between the beam and the ground
         self.revolute = chrono.ChLinkLockRevolute()
         self.revolute.Initialize(self.beam, self.ground, chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), chrono.Q_from_AngAxis(chrono.CH_C_PI / 2, chrono.ChVectorD(0, 0, 1))))#
         self.system.AddLink(self.revolute)

         # Create the motor
         positionA1 = chrono.ChVectorD(0, 0, 0) 
         self.rotmotor1 = chrono.ChLinkMotorRotationTorque()

         # Connect the rotor and the stator and add the motor to the sys:
         self.rotmotor1.Initialize(self.ground,                # body A (slave)
                               self.beam,               # body B (master)
                               chrono.ChFrameD(positionA1)  # motor frame, in abs. coords
         )
         self.system.Add(self.rotmotor1)
          
         # Create prismatic link
         self.prismatic1 = chrono.ChLinkLockPrismatic()
         self.prismatic1.Initialize(self.beam, self.body_1, chrono.ChCoordsysD(
             chrono.ChVectorD(0, 0, 0), chrono.Q_from_AngY(chrono.CH_C_PI_2)))
         self.system.AddLink(self.prismatic1)

         # Create the motor
         positionA1 = chrono.ChVectorD(0, 0, 0) 
         self.rotmotor1 = chrono.ChLinkMotorRotationTorque()

         # Connect the rotor and the stator and add the motor to the sys:
         self.rotmotor1.Initialize(self.ground,                # body A (slave)
                               self.beam,               # body B (master)
                               chrono.ChFrameD(positionA1,chrono.Q_from_AngAxis(chrono.CH_C_PI / 2, chrono.ChVectorD(0, 0, 1)))  # motor frame, in abs. coords ,
         )
         self.system.Add(self.rotmotor1)
         
         self.angle_x = 0
         
     def getState(self):
         # Return the current state
         try: 
           return np.array([self.position,self.velocity,self.beam_angle,self.beam_angle_vel])
         except:
           return np.array([0,0,0,0])
       
     def updateSystem(self, timestep = 1e-2):
         #Do step
         self.system.DoStepDynamics(timestep)
         
         #fetch current angle
         self.angle_x = shiftAngle(self.revolute.GetRelAngle())
         
         #calculate state
         self.position = self.body_1.GetPos().x/m.cos(self.angle_x)
         self.velocity = self.body_1.GetPos_dt().x/m.cos(self.angle_x)
         self.beam_angle = self.angle_x
         self.beam_angle_vel = self.revolute.GetRelWvel().z
         
         if abs(self.position)>self.beam_length / 2 and self.prismatic1.IsBroken() == False:
           self.prismatic1.SetBroken(True)
     
     def updateActuator1(self, input1):
         #update actuator1
         f_input1 = chrono.ChFunction_Const(input1)
         self.rotmotor1.SetMotorFunction(f_input1)         
           
     def updateActuator2(self, input2):
         #update actuator2
         pass              


###############################################################################