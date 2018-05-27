from __future__ import print_function
from neural_network import *
from data_set import *
import sys 
from dataCollection import *


from upperbound import upperbound

#the first way of defining paramters

if len(sys.argv) == 8: 

    if sys.argv[1] == 'mnist' or sys.argv[1] == 'cifar10': 
        dataSetName = sys.argv[1]
    else: 
        print("please specify as the 1st argumemnt the dataset: mnist or cifar10 ")
        exit

    if sys.argv[2] == 'ub' or sys.argv[2] == 'lb': 
        bound =  sys.argv[2] 
    else: 
        print("please specify as the 2nd argumemnt the bound: ub or lb ")
        exit
    
    if sys.argv[3] == 'cooperative' or sys.argv[3] == 'competitive': 
        gameType =  sys.argv[3] 
    else: 
        print("please specify as the 3nd argumemnt the game mode: cooperative or competitive ")
        exit
    
    if isinstance(int(sys.argv[4]),int): 
        image_index =  int(sys.argv[4])
    else: 
        print("please specify as the 4th argument the index of the image: [int] ")
        exit
    
    if sys.argv[5] == 'L1' or sys.argv[5] == 'L2': 
        distanceMeasure = sys.argv[5]
    else: 
        print("please specify as the 5th argument the distancemeasure: L1 or L2 ")
        exit
    
    if isinstance(float(sys.argv[6]),float): 
        distance = float(sys.argv[6])
    else: 
        print("please specify as the 6th argument the distance: [int/float] ")
        exit
    eta = (distanceMeasure,distance)
    
    if isinstance(float(sys.argv[7]),float): 
        tau = float(sys.argv[7])
    else: 
        print("please specify as the 7th argument the tau: [int/float] ")
        exit


elif len(sys.argv) == 1: 
# the second way of defining parameters
    dataSetName = 'cifar10'
    bound =  'ub'
    gameType = 'cooperative'
    image_index = 3
    eta = ("L1",40)
    tau = 0.5
    
# calling algorithms 

dc = dataCollection("%s_%s_%s_%s_%s_%s"%(dataSetName,bound,tau,gameType,image_index,eta))
dc.initialiseIndex(image_index)

if bound ==  'ub': 
    (elapsedTime, newConfident, percent, l2dist, l1dist, l0dist) = upperbound(dataSetName,bound,tau,gameType,image_index,eta)
    
    dc.addRunningTime(elapsedTime)
    dc.addConfidence(newConfident)
    dc.addManipulationPercentage(percent)
    dc.addl2Distance(l2dist)
    dc.addl1Distance(l1dist)
    dc.addl0Distance(l0dist)


else: 
    print("lower bound algorithm is developing...")
    exit
    
    
dc.provideDetails()
dc.summarise()
dc.close()


    
from keras import backend as K

K.clear_session()

