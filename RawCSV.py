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
        demoname = "user" + str(user) + "/demo" + str(demo) + ".pkl"
        data = pickle.load(open(demoname, "rb"))

        
        StateList = (data["State"])
        UserActions = (data["UserAction"])
        AutoActionList = data['AutoAction']
        InputList = (data["InputList"])

        use = StateList[3:]
        
        # for i in use:
        #     use2 = i["x"]
        #     StateInner.append(use2)
        # UserActionsTotal.append(UserActions)
        # AutoActionList.append
        InnerInputListTotal.append(InputList)

        InputTotal = 0
        #print("SIZE", (StateList[1]['x']))
        EndState = StateList[-1]
        #print("END", EndState)
        EndPos = EndState["x"][0:3]
        
        Start = (StateList[1])
        LastPos = Start["x"][0:3]


        df = pd.DataFrame(data)
        string = "User" + str(user)+"_Demo"+str(demo)+   ".csv"
        df.to_csv(string, index=False)



# with open('file.csv', 'w', newline='') as file:
#     pass

# with open('file.csv', 'w', newline='') as file:
#     writer = csv.writer(file)

# with open('file.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['name', 'age'])
#     writer.writerow(['John Doe', 30])
