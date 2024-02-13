from collections import UserList
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
endgoals = [[[.30,.350],[.30,.350],[.50,-.25],[.50,-.25],[.48,.335],[.48,.335]],#user 4 , NOTE USED OLD BLOCK SETUP
            [[.45,.275],[.45,.275],[.65,-.15],[.65,-.15],[.65,.25],[.65,.25]],#user 5
            [[.45,.275],[.45,.275],[.65,-.15],[.65,-.15],[.65,.25],[.65,.25]],#user 6
            [[.45,.275],[.45,.275],[.65,-.15],[.65,-.15],[.65,.25],[.65,.25]],#user 7 
            [[.45,.275],[.45,.275],[.65,-.15],[.65,-.15],[.65,.25],[.65,.25]],#user 8 
            [[.45,.275],[.45,.275],[.65,-.15],[.65,-.15],[.65,.25],[.65,.25]], #user 9
            [[.45,.275],[.45,.275],[.65,-.15],[.65,-.15],[.65,.25],[.65,.25]],#user 10 
            [[.45,.275],[.45,.275],[.65,-.15],[.65,-.15],[.65,.25],[.65,.25]],#user 11
            [[.45,.275],[.45,.275],[.65,-.15],[.65,-.15],[.65,.25],[.65,.25]],#user 12
            [[.45,.275],[.45,.275],[.65,-.15],[.65,-.15],[.65,.25],[.65,.25]]#user 13
            ]
DistErrorW = []
AvgDistErrorW = []
DistErrorWo = []
AvgDistErrorWo = []
userLists = [4,5,6,7,8,9,10,11,12,13]
UserDataTotal = []
percentlist = []

TotalInputWith =[]
TotalInputWO=[]

TotalTraveledListW = []
TotalTraveledListWO = []
Comms = [ 
    [2,6,4,3,5,1], #1
    [5,3,1,6,2,4], #2
    [4,2,6,5,1,3], #3
    [1,5,3,4,2,6], #4
    [3,1,5,4,6,2], #5
    [6,4,2,1,3,5], #6
    [4,3,5,1,6,2], #7
    [2,1,3,5,4,6], #8
    [6,5,1,3,2,4], #9
    [1,6,2,4,3,5], #10
    [3,2,4,6,5,1], #11
    [5,4,6,2,1,3], #12
    [1,2,3,4,5,6]  #13
] #order of use for comms vs no comms, users 1-13
StateTotal = []
UserActionsTotal =[]
AutoActionList = []
InputListTotal = []
count =0
for user in userLists:
    demo = 0
    count +=1

    InputWith =[0]*3
    InputWO=[0]*3

    print("User", user)
    InputData = [0]*6
    List = Comms[user-1]
    W_Comm = 0
    Wo_Comm = 0
    ErrorUser = [0]*6
    GoalSet = endgoals[count-1]

    InnerInputListTotal = []
    StateInner = []

    TraveledListW = []
    TraveledListWO = []

    for i in range(6):
        demo += 1
        DistTraveled = 0
        Setup = List[demo-1]
        Goal = GoalSet[Setup-1]
        #print("Cycle Demo", demo,"Setup ", Setup )
        demoname = "data/user" + str(user) + "/demo" + str(demo) + ".pkl"
        data = pickle.load(open(demoname, "rb"))

        
        StateList = (data["State"])
        UserActions = (data["UserAction"])
        AutoActionList = data['AutoAction']
        InputList = (data["InputList"])

        use = StateList[1:]
        for i in use:
            temp = i["x"]
            StateInner.append(temp[0:3])
        
        # UserActionsTotal.append(UserActions)
        # AutoActionList.append
        InnerInputListTotal.append(InputList)
        #TotalTime = data["TotalTime"] #FUCKED UP SAVING THIS, THIS IS INACCURATE
        #Timestep = TotalTime/len(InputList)
        #print("TIMEY,TIMESTEP",Timestep,TotalTime)
        InputTotal = 0
        #print("SIZE", (StateList[1]['x']))
        EndState = StateList[-1]
        #print("END", EndState)
        EndPos = EndState["x"][0:3]
        
        Start = (StateList[1])
        LastPos = Start["x"][0:3]
        #Limiting data to when goal is reached
        for j in range(len(StateList)-1):
            #print(StateList[j])
            State = StateList[j+1]['x']
            #print("BUBBA",State)
            Pos = State[0:3]
            DistTraveled += np.linalg.norm(Pos-LastPos)
            LastPos = Pos
            if np.linalg.norm(Pos-EndPos) < .1:
                #print("FREEDOM",j)
                break
            
        

        
        InputList = InputList[0:j]
        #Input Mag
        for j in range(len(InputList)):
            InputTotal += np.sum(np.abs(InputList[j]))
        #Input Count
        # for j in range(len(InputList)):
        #     if np.sum(np.abs(InputList[j])) > .25:
        #         InputTotal += 1
        InputData[Setup-1] = InputTotal
        ErrorUser[Setup-1] = np.linalg.norm(EndPos[0:2]-Goal)
        #print("GOAL:",Goal,"---- End:",EndPos)
        if Setup == 2 or Setup == 4 or Setup == 6:
            W_Comm += InputTotal
            DistErrorW.append(ErrorUser)
            AvgDistErrorW.append(np.mean(ErrorUser))
            TraveledListW.append(DistTraveled)
            if Setup == 2:
                InputWith[0] = InputTotal
            if Setup == 4:
                InputWith[1] = InputTotal
            if Setup == 6:
                InputWith[2] = InputTotal
            
            #print("User ", user,"-",demo, "-",Setup, " ,Input Total of ",InputTotal, "----- End State is at, " ,EndPos)
        if Setup == 1 or Setup == 5 or Setup == 3:
            Wo_Comm += InputTotal
            DistErrorWo.append(ErrorUser)
            AvgDistErrorWo.append(np.mean(ErrorUser))
            TraveledListWO.append(DistTraveled)
            #print("User ", user ,"-",demo,"-",Setup, " ,Input Total of ",InputTotal, "----- End State is at, " ,EndPos)
            if Setup == 1:
                InputWO[0] = InputTotal
            if Setup == 3:
                InputWO[1] = InputTotal
            if Setup == 5:
                InputWO[2] = InputTotal

    TotalInputWith.append(InputWith) 
    TotalInputWO.append(InputWO) 
    TotalTraveledListW.append(TraveledListW)
    TotalTraveledListWO.append(TraveledListWO)
    StateTotal.append(StateInner)
    InputListTotal.append(InnerInputListTotal)
        #print()
    percentage = ((Wo_Comm/W_Comm)-1 )*100
    if Wo_Comm > W_Comm:
        print("Success W_Comm required ", ((Wo_Comm/W_Comm)-1 )*100, " percent less inputs" )
        
    else:
        print("FAILURES --- W_Comm required ", ((W_Comm/Wo_Comm)-1 )*100, " percent more inputs" )
    print("AVG DIST ERROR", np.mean(ErrorUser))
    
    UserDataTotal.append(InputData)
    percentlist.append(percentage)
    
print("Total Average Difference in Input Percentage",(np.sum(percentlist))/len(percentlist),"Number tested:",len(percentlist))
print("Total Average Difference in Distance",np.abs((np.mean(AvgDistErrorW)-np.mean(AvgDistErrorWo))),"Number tested:",len(percentlist))

## PLOTTING ---------
#input numbers
#get metrics

# sum1 = 0
# sum2 = 0
# sum3 = 0
# sum1b = 0
# sum2b = 0
# sum3b = 0
# for i in range(len(TotalInputWith)):
#     InputWithUse = TotalInputWith[i]
#     InputWOUse = TotalInputWO[i]
    
#     sum1 += InputWithUse[0]
#     sum1b += InputWOUse[0]
#     sum2 += InputWithUse[1]
#     sum2b += InputWOUse[1]
#     sum3 += InputWithUse[2]
#     sum3b += InputWOUse[2]

# avg1 = sum1/len(TotalInputWith)
# avg1b = sum1b/len(TotalInputWith)
# avg2 = sum2/len(TotalInputWith)
# avg2b = sum2b/len(TotalInputWith)
# avg3 = sum3/len(TotalInputWith)
# avg3b = sum3b/len(TotalInputWith)

# mean = [avg1b,avg1,avg2b,avg2,avg3,avg3b]
# colorwheel = {'Orange':[255/256, 153/256, 0], 'Green':[160/256, 212/256, 164/256], 'Blue':[42/256, 143/256, 189/256],'Purple': [141/256, 95/256, 211/256],
#               'Light Gray': [179/256, 179/256, 179/256],'Dark Gray':[102/256, 102/256, 102/256]} #Orange, Green, Blue, Purple, Light Gray, Dark Gray


# # plot result
# x = ["Env 1 W/","Env 1 W/o","Env 2 W/","Env 2 W/o","Env 3 W/","Env 3 W/o"]
# plt.bar(x, [avg1b,avg1,avg2b,avg2,avg3b,avg3],color=[colorwheel['Orange'],colorwheel['Green'],colorwheel['Orange'],colorwheel['Green'],colorwheel['Orange'],colorwheel['Green']])
# #plt.errorbar(x, mean, sem)
# #plt.legend(["Without Feedback","With Feedback"])
# plt.title("Total User Inputs")
# plt.show()

# #Travel Plots
# Travelsum1 = 0
# Travelsum2 = 0
# Travelsum3 = 0
# Travelsum1b = 0
# Travelsum2b = 0
# Travelsum3b = 0
# for i in range(len(TotalTraveledListW)):
#     TravelWithUse = TotalTraveledListW[i]
#     TravelWOUse = TotalTraveledListWO[i]
    
#     Travelsum1 += TravelWithUse[0]
#     Travelsum1b += TravelWOUse[0]
#     Travelsum2 += TravelWithUse[1]
#     Travelsum2b += TravelWOUse[1]
#     Travelsum3 += TravelWithUse[2]
#     Travelsum3b += TravelWOUse[2]

# Travelavg1 = Travelsum1/len(TotalInputWith)
# Travelavg1b = Travelsum1b/len(TotalInputWith)
# Travelavg2 = Travelsum2/len(TotalInputWith)
# Travelavg2b = Travelsum2b/len(TotalInputWith)
# Travelavg3 = Travelsum3/len(TotalInputWith)
# Travelavg3b = Travelsum3b/len(TotalInputWith)

# #TavelPlots
# x = ["Env 1 W/","Env 1 W/o","Env 2 W/","Env 2 W/o","Env 3 W/","Env 3 W/o"]
# plt.bar(x, [Travelavg1b,Travelavg1,Travelavg2b,Travelavg2,Travelavg3b,Travelavg3],color=[colorwheel['Light Gray'],colorwheel['Green'],colorwheel['Light Gray'],colorwheel['Green'],colorwheel['Light Gray'],colorwheel['Green']])
# #plt.errorbar(x, mean, sem)
# #plt.legend(["Without Feedback","With Feedback"])
# plt.title("Distance Traveled")
# plt.show()

# #Dist Plots
# plt.bar(["Dist W/ Feedback","Dist W/O Feedback"],[np.mean(AvgDistErrorW),np.mean(AvgDistErrorWo)],color=[colorwheel['Orange'],colorwheel['Green']])
# plt.title("Average Final Distance from Goal")
# #plt.ylim([0,np.mean([avg1b,avg1,avg2b,avg2,avg3b,avg3])])
# #plt.legend(["Without Feedback","With Feedback"])
# plt.show()



# import pandas as pd

data = {
    'InputsWith':TotalInputWith,
    'InputsWithOut':TotalInputWO,
    }

data2 = {
    'AvgDistanceFromGoal_W':AvgDistErrorW,
    'AvgDistanceFromGoal_WO':AvgDistErrorWo,
}

data3 = {
    'Travel_Distance_W':TotalTraveledListW,
    'Travel_Distance_W':TotalTraveledListWO,
}




df = pd.DataFrame(data)
df.to_csv('InputsTotalData.csv', index=False)

df2 = pd.DataFrame(data2)
df2.to_csv('AvgDistanceFromGoal.csv', index=False)

df3 = pd.DataFrame(data3)
df3.to_csv('TraveledDistance.csv', index=False)


# with open('file.csv', 'w', newline='') as file:
#     pass

# with open('file.csv', 'w', newline='') as file:
#     writer = csv.writer(file)

# with open('file.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['name', 'age'])
#     writer.writerow(['John Doe', 30])
