#from collections import UserList
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
import pandas as pd



def data2array(name):
    data = pickle.load(open(name, "rb"))

    datalist = []
    for item in data:
        row = []
        for array in item:
            row.append(list(array))
        datalist.append(row)
    return np.array(data)

def get_error(data):
    error = np.zeros((len(data), 8))
    for idx, item in enumerate(data):
        theta = item[0]
        for jdx in range(8):
            error[idx, jdx] = np.linalg.norm(theta - item[jdx+1])
    return list(error)



#user = 9 #Good Users 4,6,7,8,9,10,11,12,13 Alright Users 1,6


userLists = [4,8,11,12,6]
UserDataTotal = []
percentlist = []
percentlist2 = []

TotalInputMethod = []
TotalInputWith =[]
TotalInputWO=[]

TotalTaskInputMethod = []
TotalTaskInputWith =[]
TotalTaskInputWO=[]

TotalTraveledListMethod = []
TotalTraveledListJavW = []
TotalTraveledListJavWO = []

TotalTimeData_Method = []
TotalTimeData_WJav = []
TotalTimeData_WOJav = []

Comms = [ 
    [1,2,3,4,5,6,7,8,9], #1
    [1,2,3,4,5,6,7,8,9], #2
    [1,2,3,4,5,6,7,8,9], #3
    [1,2,3,4,5,6,7,8,9], #4
    [1,2,3,4,5,6,7,8,9], #5
    [1,2,3,4,5,6,7,8,9], #6
    [1,2,3,4,5,6,7,8,9], #7
    [1,2,3,4,5,6,7,8,9], #8
    [1,2,3,4,5,6,7,8,9], #9
    [1,2,3,4,5,6,7,8,9], #10
    [1,2,3,4,5,6,7,8,9], #11
    [1,2,3,4,5,6,7,8,9], #12
    [1,2,3,4,5,6,7,8,9]  #13
]
        
salt   = [-0.784, -0.1825, 0.25]
pepper = [-0.634, -0.260, 0.255]

plate = [-0.555, -0.17, 0.3]
fork  = [-0.321, -0.198, 0.325]
spoon = [-0.445, -0.288, 0.325]

fork2  = [-0.365, 0.195, 0.28]
spoon2 = [-0.730, 0.205, 0.28]

can = [-0.315, -0.035, 0.3]
mug = [-0.650,-0.055, 0.3]
#Salt,pepper,plate,fork, spoon, fork2, spoon2, cup, mug
EnvGoals = [salt,pepper,plate,fork,spoon,fork2,spoon2,can,mug]
goalset2 = [[0,2,0],[7,8,7],[4,6],[0,2,0],[7,8,7],[4,6],[0,2,0],[7,8,7],[4,6]]

goalset1 = [[1,2,1],[7,8,7],[3,5],[1,2,1],[7,8,7],[3,5],[1,2,1],[7,8,7],[3,5]]
group1 = [1,3,5,8,10,12]
group2 = [2,4,6,7,9 ,11]
groupset = [1,2,1,2,1,2, 2,1,2,1,2,1]
count =0
for user in userLists:
    demo = 0
    count +=1
    InputJavWith =[0]*3
    InputJavWO=[0]*3
    InputMethod=[0]*3
    if groupset[user-1] == 1:
        goalset = goalset1
        print("Group 1")
    else:
        if groupset[user-1] == 2:
            goalset = goalset2
            print("Group 2")
        else:
            print("FAILURE")   
    print("User", user)
    InputData = [0]*9
    List = Comms[user-1]
    UserTotalInput_WJav = 0
    UserTotalInput_WOJav = 0
    UserTotalInput_Method = 0

    UserActionDiff = [0]*9
    TotalActionDiff = []

    TraveledDist_WJav = []
    TraveledDist_WOJav = []
    TraveledDist_Method = []

    UserTimeData_Method = [0]*3
    UserTimeData_WJav = [0]*3
    UserTimeData_WOJav = [0]*3
    for i in range(9):
        goals = goalset[demo]
        numgoals = len(goals)
        currgoal = 0
        #print(numgoals)
        demo += 1
        
        DistTraveled = 0
        Setup = List[demo-1]
        #print("Cycle Demo", demo,"Setup ", Setup )
        demoname = "data/user" + str(user) + "/demo" + str(demo) + ".pkl"
        data = pickle.load(open(demoname, "rb"))

        StateList = (data["State"])
        UserActions = (data["UserAction"])
        AutoActionList = data['AutoAction']
        InputList = (data["InputList"])
        TimeList = (data["TotalTime"])

        TimeTaken = 0
        InputTotal = 0
        ActionDiff = 0
        #print("SIZE", (StateList[1]['x']))
        EndState = StateList[-1]
        #print("END", EndState)
        EndPos = EndState["x"][0:3]
        Start = (StateList[1])
        LastPos = Start["x"][0:3]
        #Limiting data to when goal is reached
        for j in range(len(StateList)-1):
            #print(StateList[j])
            cycled = False
            State = StateList[j+1]['x']
            #print("BUBBA",State)
            Pos = np.array(State[0:3])
            DistTraveled += np.linalg.norm((Pos)-(LastPos))
            LastPos = Pos     
            # Robot = np.array(AutoActionList[j])
            # Robot = Robot[0]

            # User = np.array(UserActions[j])
            # User = User[0]
            # if np.linalg.norm(Robot) < .005:
            #     Robot_Action = Robot
            # else:
            #     Robot_Action = Robot / np.linalg.norm(Robot)
            #     #print(Robot,np.linalg.norm(Robot))
            # if np.linalg.norm(User) < .005:
            #     User_Action = Robot
            # else:
            #     User_Action = User / np.linalg.norm(User)
            
            #ActionDiff += np.linalg.norm(User_Action-Robot_Action) 
            if user == 1:
                TimeTaken += TimeList[j]    
            if user == 2:
                if Setup > 6:
                    TimeTaken += TimeList[j]    
            if user == 3:
                if Setup > 6:
                    TimeTaken += TimeList[j]  
        #ActionDiff /= j
#
        for j in range(len(InputList)):
            InputTotal += np.sum(np.abs(InputList[j]))
        #Input Count
        # for j in range(len(InputList)):
        #     if np.sum(np.abs(InputList[j])) > .25:
        #         InputTotal += 1.0
                

        InputData[Setup-1] = InputTotal
        if user == 1:
            TimeTaken += TimeList[0] 
        else:
            if user == 2 or user == 3:
                if Setup>6:
                    TimeTaken += TimeList[0] 
                else:
                    TimeTaken = TimeList[-1]
            else:
                TimeTaken = TimeList[-1]
        
        #print("TimeTaken",TimeTaken)
        #UserActionDiff[Setup-1] = ActionDiff
        if Setup == 4 or Setup == 5 or Setup == 6:
            UserTotalInput_WJav += InputTotal
            TraveledDist_WJav.append(DistTraveled)
            if Setup == 4:
                InputJavWith[0] = InputTotal
                UserTimeData_WJav[0] = TimeTaken
            if Setup == 5:
                InputJavWith[1] = InputTotal
                UserTimeData_WJav[1] = TimeTaken
            if Setup == 6:
               InputJavWith[2] = InputTotal
               UserTimeData_WJav[2] = TimeTaken
            
            #print("User ", user,"-",demo, "-",Setup, " ,Input Total of ",InputTotal, "----- End State is at, " ,EndPos)
        if Setup == 1 or Setup == 2 or Setup == 3:
            UserTotalInput_WOJav += InputTotal
            TraveledDist_WOJav.append(DistTraveled)
            #print("User ", user ,"-",demo,"-",Setup, " ,Input Total of ",InputTotal, "----- End State is at, " ,EndPos)
            if Setup == 1:
                InputJavWO[0] = InputTotal
                UserTimeData_WOJav[0] = TimeTaken
            if Setup == 2:
                InputJavWO[1] = InputTotal
                UserTimeData_WOJav[1] = TimeTaken
            if Setup == 3:
                InputJavWO[2] = InputTotal
                UserTimeData_WJav[2] = TimeTaken

        if Setup ==7 or Setup == 8 or Setup == 9:
            UserTotalInput_Method += InputTotal
            TraveledDist_Method.append(DistTraveled)
            #print("User ", user ,"-",demo,"-",Setup, " ,Input Total of ",InputTotal, "----- End State is at, " ,EndPos)
            if Setup == 7:
                InputMethod[0] = InputTotal
                UserTimeData_Method[0] = TimeTaken
            if Setup == 8:
                InputMethod[1] = InputTotal
                UserTimeData_Method[1] = TimeTaken
            if Setup == 9:
                InputMethod[2] = InputTotal
                UserTimeData_Method[2] = TimeTaken
        #UserActionDiff[demo-1] = ActionDiff
        #print(UserActionDiff)
        print("User",user,demo,InputTotal)
    TotalInputMethod.append(UserTotalInput_Method) 
    TotalInputWith.append(UserTotalInput_WJav) 
    TotalInputWO.append(UserTotalInput_WOJav) 

    TotalTaskInputMethod.append(InputMethod) 
    TotalTaskInputWith.append(InputJavWith) 
    TotalTaskInputWO.append(InputJavWO) 

    TotalTraveledListMethod.append(TraveledDist_Method)
    TotalTraveledListJavW.append(TraveledDist_WJav)
    TotalTraveledListJavWO.append(TraveledDist_WOJav)

    TotalTimeData_WOJav.append(UserTimeData_WOJav)
    TotalTimeData_WJav.append(UserTimeData_WJav)
    TotalTimeData_Method.append(UserTimeData_Method)

    #TotalActionDiff.append(UserActionDiff)
    print("USER",user,demo,UserTotalInput_Method,UserTotalInput_WJav)
    percentage = (1.0-(UserTotalInput_Method/UserTotalInput_WJav)) 
    # percentage2 = ((UserTotalInput_Method/UserTotalInput_WOJav)-1 )
    # percentage3 = ((UserTotalInput_WJav/UserTotalInput_WOJav)-1 )
    if UserTotalInput_Method < UserTotalInput_WJav:
        print("Success our Method required ", percentage*100, " percent less inputs" )
        print("Our method travelled ", (np.sum(TraveledDist_Method)/np.sum(TraveledDist_WJav))*100,"of the Distance of the baseline")
        #print((np.sum(TraveledDist_Method)),(np.sum(TraveledDist_WJav)) )
        print("Our method required ", (np.sum(UserTimeData_Method)/np.sum(UserTimeData_WJav))*100,"of the time of the baseline")
        #print( (np.sum(UserTimeData_Method)),np.sum(UserTimeData_WJav))
        #print("Action Diff = ", np.mean(UserActionDiff[3:6]),np.mean(UserActionDiff[6:9]))
        

    
    UserDataTotal.append(InputData)
    percentlist.append(percentage)
   # percentlist2.append([percentage,percentage2,percentage3])
    
#print("Total Average Difference in Input Percentage",(np.sum(percentlist))/len(percentlist),"Number tested:",len(percentlist))

## PLOTTING ---------
#input numbers
#get metrics

sum1 = 0
sum2 = 0
sum3 = 0
sum1b = 0
sum2b = 0
sum3b = 0
sum1c = 0
sum2c = 0
sum3c = 0

for i in range(len(TotalTaskInputWith)):
    InputWithUse = TotalTaskInputWith[i]
    InputWOUse = TotalTaskInputWO[i]
    InputMethod = TotalTaskInputMethod[i]
    
    sum1 += InputWithUse[0]
    sum1b += InputWOUse[0]
    sum1c += InputMethod[0]

    sum2 += InputWithUse[1]
    sum2b += InputWOUse[1]
    sum2c += InputMethod[1]

    sum3 += InputWithUse[2]
    sum3b += InputWOUse[2]
    sum3c += InputMethod[2]

    
    
    
    

avg1 = sum1/len(TotalTaskInputWith)
avg1b = sum1b/len(TotalTaskInputWith)
avg1c = sum1c/len(TotalTaskInputWith)

avg2 = sum2/len(TotalTaskInputWith)
avg2b = sum2b/len(TotalTaskInputWith)
avg2c = sum2c/len(TotalTaskInputWith)

avg3 = sum3/len(TotalTaskInputWith)
avg3b = sum3b/len(TotalTaskInputWith)
avg3c = sum3c/len(TotalTaskInputWith)

#print("NUM",len(TotalTaskInputWith))
mean = [avg1b,avg1,avg1c,avg2b,avg2,avg2c,avg3b,avg3,avg3c]
colorwheel = {'Orange':[255.0/256.0, 153.0/256.0, 0.0], 'Green':[160.0/256.0, 212.0/256.0, 164.0/256.0], 'Blue':[42.0/256.0, 143.0/256.0, 189.0/256.0],'Purple': [141.0/256.0, 95.0/256.0, 211.0/256.0],
              'Light Gray': [179.0/256.0, 179.0/256.0, 179.0/256.0],'Dark Gray':[102.0/256.0, 102.0/256.0, 102.0/256.0]} #Orange, Green, Blue, Purple, Light Gray, Dark Gray
colorset = [colorwheel['Green'],colorwheel['Orange'],colorwheel['Blue'],colorwheel['Green'],colorwheel['Orange'],colorwheel['Blue'],colorwheel['Green'],colorwheel['Orange'],colorwheel['Blue']]
a = [255/256, 153/256, 0.0]
# plot result
x = range(9)
plt.bar(x, mean,color=[colorwheel['Green'],colorwheel['Orange'],colorwheel['Blue'],colorwheel['Green'],colorwheel['Orange'],colorwheel['Blue'],colorwheel['Green'],colorwheel['Orange'],colorwheel['Blue']])
#plt.errorbar(x, mean, sem)
#plt.legend(["Javdani(WITHOUT)","JAVDANI(WITH)","Method(With)"])
plt.title("Total User Inputs")
plt.show()

#Travel Plots
Travelsum1 = 0
Travelsum2 = 0
Travelsum3 = 0

Travelsum1b = 0
Travelsum2b = 0
Travelsum3b = 0

Travelsum1c = 0
Travelsum2c = 0
Travelsum3c = 0
for i in range(len(TotalTraveledListJavWO)):
    TravelWithUse = TotalTraveledListJavW[i]
    TravelWOUse = TotalTraveledListJavWO[i]
    TravelMethod = TotalTraveledListMethod[i]
    
    
    Travelsum1 += TravelWithUse[0]
    Travelsum1b += TravelWOUse[0]
    Travelsum1c += TravelMethod[0]

    Travelsum2 += TravelWithUse[1]
    Travelsum2b += TravelWOUse[1]
    Travelsum2c += TravelMethod[1]

    Travelsum3 += TravelWithUse[2]
    Travelsum3b += TravelWOUse[2]
    Travelsum3c += TravelMethod[2]

Travelavg1 = Travelsum1/len(TotalInputWith)
Travelavg1b = Travelsum1b/len(TotalInputWith)
Travelavg1c = Travelsum1c/len(TotalInputWith)

Travelavg2 = Travelsum2/len(TotalInputWith)
Travelavg2b = Travelsum2b/len(TotalInputWith)
Travelavg2c = Travelsum2c/len(TotalInputWith)

Travelavg3 = Travelsum3/len(TotalInputWith)
Travelavg3b = Travelsum3b/len(TotalInputWith)
Travelavg3c = Travelsum3c/len(TotalInputWith)

#TavelPlots
mean = [Travelavg1b,Travelavg1,Travelavg1c,Travelavg2b,Travelavg2,Travelavg2c,Travelavg3b,Travelavg3,Travelavg3c]
x = range(9)
# plt.bar(x,mean ,color = colorset)
#plt.errorbar(x, mean, sem)
#plt.legend(["Without Feedback","With Feedback"])
# plt.title("Distance Traveled")
# plt.show()

#Travel Plots
Timesum1 = 0
Timesum2 = 0
Timesum3 = 0

Timesum1b = 0
Timesum2b = 0
Timesum3b = 0

Timesum1c = 0
Timesum2c = 0
Timesum3c = 0

for i in range(len(TotalTraveledListJavWO)):
    TimeWithUse = TotalTimeData_WJav[i]
    TimeWOUse = TotalTimeData_WOJav[i]
    TimeMethod = TotalTimeData_Method[i]
    
    
    Timesum1 += TimeWithUse[0]
    Timesum1b += TimeWOUse[0]
    Timesum1c += TimeMethod[0]

    Timesum2 += TimeWithUse[1]
    Timesum2b += TimeWOUse[1]
    Timesum2c += TimeMethod[1]

    Timesum3 += TimeWithUse[2]
    Timesum3b += TimeWOUse[2]
    Timesum3c += TimeMethod[2]

Timeavg1 = Timesum1/len(TotalTimeData_WJav)
Timeavg1b = Timesum1b/len(TotalTimeData_WJav)
Timeavg1c = Timesum1c/len(TotalTimeData_WJav)

Timeavg2 = Timesum2/len(TotalTimeData_WJav)
Timeavg2b = Timesum2b/len(TotalTimeData_WJav)
Timeavg2c = Timesum2c/len(TotalTimeData_WJav)

Timeavg3 = Timesum3/len(TotalTimeData_WJav)
Timeavg3b = Timesum3b/len(TotalTimeData_WJav)
Timeavg3c = Timesum3c/len(TotalTimeData_WJav)

#TavelPlots
mean = [Timeavg1b,Timeavg1,Timeavg1c,Timeavg2b,Timeavg2,Timeavg2c,Timeavg3b,Timeavg3,Timeavg3c]
x = range(9)
# plt.bar(x,mean ,color = colorset)
# #plt.errorbar(x, mean, sem)
# #plt.legend(["Without Feedback","With Feedback"])
# plt.title("Time per task trial")
# plt.show()
