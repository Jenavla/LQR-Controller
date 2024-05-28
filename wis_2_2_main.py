# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:33:49 2023

@author: Rik
"""

import wis_2_2_utilities as util
import wis_2_2_systems as systems
#import wis_2_2_utilities_nochrono as util
#import wis_2_2_systems_nochrono as systems
import matplotlib.pyplot as plt
import pandas as pd
import control as ct
import numpy as np

# set constants
timestep = 2e-3

Q = np.array([[5e6, 0, 0, 0, 0, 0],
              [0, 5.7e5, 0, 0, 0, 0],
              [0, 0, 0.1, 0, 0, 0],
              [0, 0, 0, 1.9, 0, 0],
              [0, 0, 0, 0, 0.1, 0],
              [0, 0, 0, 0, 0, 1.9]])
R = np.array([[1]])
matrix_A = np.array([[0,1,0,0,0,0],
                     [0,0,-1.82816363e-01,0,-1.86280695e-01,0],
                     [0,0,0,1,0,0],
                     [0,0,2.56871779e+01,0,-2.47278963e+01,0],
                     [0,0,0,0,0,1],
                     [0,0,-3.42360461e+01,0,9.17711190e+01,0]]) 
matrix_B = np.array([[0],
                    [8.87313309e-02],
                    [0],
                    [-2.28090355e-01],
                    [0],
                    [3.04001902e-01]]) 
matrix_C = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1]])

matrix_K = ct.lqr(matrix_A,matrix_B,Q,R)[0]




class controller():
    def __init__(self, target=0):
        #initial estimate for the state:
        self.state_estimate = np.asarray([[0.],[0],[0],[0],[0],[0]])
        self.u = np.asarray([[0.]])
        list_poles_estimate = list_poles_feedback = [-3,-6,-9,-12,-15,-18]
            
        self.matrix_gain=np.array([matrix_K])
        
        #construct gain matrix by pole placement
        self.matrix_gain_feedback=ct.place(matrix_A,matrix_B,list_poles_feedback)
        self.matrix_gain_estimate=ct.place(matrix_A.T,matrix_C.T,list_poles_estimate).T

    def updateEstimate(self, observe):
     
        y = matrix_C @ np.array([observe]).T
        y_hat = matrix_C @ self.state_estimate
        
        self.state_estimate += timestep*(matrix_A @ self.state_estimate\
                                       + matrix_B @ self.u\
                                       + self.matrix_gain_estimate @ (y-y_hat))
  
  
  
    def feedBack(self, observe):
        self.updateEstimate(observe)
        
        self.u = -self.matrix_gain @ observe
        #self.u = -self.matrix_gain_feedback @ np.array([observe]).T
        return self.u
        
def datawork():
    """
    Created on Thur 16-11-2023 15:24

    @author: Nathan
    """
    eigenvalues = np.linalg.eigvals(matrix_A)
    
    # Controllability
    wc = ct.ctrb(matrix_A, matrix_B)
    controllability_rank = np.linalg.matrix_rank(wc)
    print("Controleerbaarheid:", "Volledig controleerbaar" if controllability_rank == matrix_A.shape[0] else "Niet volledig controleerbaar")

    # Stabilizability
    uncontrollable_modes = eigenvalues[controllability_rank:]
    print("Stabiliseerbaarheid:", "Stabiliseerbaar" if all(np.real(uncontrollable_modes) < 0) else "Niet stabiliseerbaar")

    # Observability
    wo =ct.obsv(matrix_A, matrix_C)
    observability_rank = np.linalg.matrix_rank(wo)
    print("Observeerbaarheid:", "Volledig observeerbaar" if observability_rank == matrix_A.shape[0] else "Niet volledig observeerbaar")

    # Detectability
    unobservable_modes = eigenvalues[observability_rank:]
    print("Detecteerbaarheid:", "Detecteerbaar" if all(np.real(unobservable_modes) > 0) else "Niet detecteerbaar")
    
    # Read the data
    data = pd.read_csv('cart_inverted_pendulum.csv', sep=',')
    
    # Define your column names
    column_names = ['tijd', 'kwad_toestand_kosten', 'kwad_input_kosten', 'positie kar','snelheid kar','hoek_slinger_1','hoeksnelheid slinger 1','hoek slinger 2', 'hoeksnelheid slinger 2','input'] 
    
    # Assign the column names to the DataFrame
    data.columns = column_names
    
    tijd = data['tijd']
    toestand_kosten = data['kwad_toestand_kosten']
    input_kosten = data['kwad_input_kosten']
    positie_kar = data['positie kar']
    snelheid_kar = data['snelheid kar']
    hoek_1 = data['hoek_slinger_1']
    hoeksnelheid_1 = data['hoeksnelheid slinger 1']
    hoek_2 = data['hoek slinger 2']
    hoeksnelheid_2 = data['hoeksnelheid slinger 2']
    inputs = data['input']
    
    # Find point where system became stable 
    snijpunten_1 = []
    snijpunten_2 = []
    snijpunten_3 = []
    snijpunten_1.append([0.0, 0.0])
    snijpunten_2.append([0.0, 0.0])
    snijpunten_3.append([0.0, 0.0])    
    for i in range(1, len(tijd)):
        if (-0.005 >= positie_kar[i] or 0.005 <= positie_kar[i]):
            snijpunten_1.append([tijd[i], positie_kar[i]])
        if (-0.1 >=hoeksnelheid_1[i] or 0.1 <= hoeksnelheid_1[i]):
            snijpunten_2.append([tijd[i], hoeksnelheid_1[i]])
        if (-0.1 >=hoeksnelheid_2[i] or 0.1 <= hoeksnelheid_2[i]):
            snijpunten_3.append([tijd[i], hoeksnelheid_2[i]])
    
    print("Toestand kosten:",max(toestand_kosten))
    print("Sijpunten Positie kar:",snijpunten_1[-1])
    print("Sijpunten Pendulum 1:",snijpunten_2[-1])
    print("Sijpunten Pendulum 2:",snijpunten_3[-1])
    
    #plot
    plt.plot(tijd, inputs)
    plt.xlabel('Tijd(s)')
    plt.ylabel('Input')
    plt.title('Inputkosten over tijd')
    plt.show()
    
    #plot
    plt.plot(tijd, positie_kar)
    plt.xlabel('Tijd(s)')
    plt.ylabel('positie kar')
    plt.title('Positie kar over tijd')
    plt.scatter(snijpunten_1[-1][0], snijpunten_1[-1][1], color= 'g')
    plt.axhline(y=0.005, color='red', linestyle='--', label='0.005')
    plt.axhline(y=-0.005, color='red', linestyle='--', label='-0.005')
    # Set y-axis limits
    plt.ylim(-0.10, 0.10) 
    plt.show()
        
    #plot
    plt.plot(tijd, toestand_kosten)
    plt.xlabel('Tijd(s)')
    plt.ylabel('toestand kosten')
    plt.title('toestand kosten over tijd')
    # Set y-axis limits
    plt.show()
    
    plt.plot(tijd, snelheid_kar)
    plt.xlabel('Tijd(s)')
    plt.ylabel('Snelheid kar')
    plt.title('Snelheid kar over tijd')
    plt.show()

    #plot
    plt.plot(tijd, hoek_1)
    plt.xlabel('Tijd(s)')
    plt.ylabel('Positie slinger 1')
    plt.title('Positie over tijd van Pendulum 1')
    plt.show()
    
    #plot
    plt.plot(tijd, hoeksnelheid_1)
    plt.xlabel('Tijd(s)')
    plt.ylabel('Snelheid slinger 1')
    plt.title('Snelheid over tijd van Pendulum 1')
    plt.scatter(snijpunten_2[-1][0], snijpunten_2[-1][1], color= 'g')
    plt.axhline(y=0.1, color='red', linestyle='--', label='0.1')
    plt.axhline(y=-0.1, color='red', linestyle='--', label='-0.1')
    # Set y-axis limits
    plt.ylim(-0.15, 0.15) 
    plt.show()
    
    #plot
    plt.plot(tijd, hoek_2)
    plt.xlabel('Tijd(s)')
    plt.ylabel('Positie slinger 2')
    plt.title('Positie over tijd van Pendulum 2')
    plt.show()

    plt.plot(tijd, hoeksnelheid_2, label='Actual Snelheid slinger 2')
    plt.xlabel('Tijd(s)')
    plt.ylabel('Snelheid slinger 2')
    plt.title('Snelheid over tijd van Pendulum 2')
    plt.scatter(snijpunten_3[-1][0], snijpunten_3[-1][1], color= 'g')
    plt.axhline(y=0.1, color='red', linestyle='--', label='0.1')
    plt.axhline(y=-0.1, color='red', linestyle='--', label='-0.1')
    # Set y-axis limits
    plt.ylim(-0.15, 0.15) 
    plt.show()
      
    #plot
    plt.plot(tijd, toestand_kosten)
    plt.xlabel('Tijd(s)')
    plt.ylabel('Toestandkosten')
    plt.title('Toestand_kosten over tijd van Pendulum 2', print(max(toestand_kosten)))
    plt.show()

def main():
  model=systems.cart_inverted_pendulum(pendulum1_length = 0.5, pendulum2_length = 0.5, 
                                       second_pendulum = True, high_kick=5)
  control = controller()
  simulation = util.simulation(model=model,timestep=timestep)
  simulation.setCost()
  simulation.max_duration = 5 #seconde
  simulation.GIF_toggle = True #set to false to avoid frame and GIF creation

  while simulation.vis.Run():
      if simulation.time<simulation.max_duration:
        simulation.step()
        u = control.feedBack(simulation.observe())
        simulation.control(u)
        #simulation.logEstimate(control.state_estimate)
        simulation.log()
        simulation.refreshTime()
      else:
        print('Ending visualisation...')
        simulation.vis.GetDevice().closeDevice()
        
  simulation.writeData()
  datawork()
    
  
if __name__ == "__main__":
 main()
 