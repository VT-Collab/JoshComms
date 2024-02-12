import numpy as np
import pickle
#import matplotlib.pyplot as plt


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



user = 4

demo = 0
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
print("User", user)
InputData = [0]*6
List = Comms[user-1]
W_Comm = 0
Wo_Comm = 0
for i in range(6):
    demo += 1

    Setup = List[demo-1]
    #print("Cycle Demo", demo,"Setup ", Setup )
    demoname = "data/user" + str(user) + "/demo" + str(demo) + ".pkl"
    data = pickle.load(open(demoname, "rb"))

    
    StateList = (data["State"])
    UserActions = (data["UserAction"])
    AutoActionList = data['AutoAction']
    InputList = (data["InputList"])
    InputTotal = 0
    #print("SIZE", (StateList[1]['x']))
    EndState = StateList[-1]
    #print("END", EndState)
    EndPos = EndState["x"][0:3]
    
    #Limiting data to when goal is reached
    for j in range(len(StateList)-1):
        #print(StateList[j])
        State = StateList[j+1]['x']
        #print("BUBBA",State)
        Pos = State[0:3]
        
        if np.linalg.norm(Pos-EndPos) < .1:
            #print("FREEDOM",j)
            break


    
    InputList = InputList[0:j]
    for j in range(len(InputList)):
        InputTotal += np.sum(np.abs(InputList[j]))
        #print((np.abs(InputList[j])),InputList[j])
    InputData[Setup-1] = InputTotal
    if Setup == 2 or Setup == 4 or Setup == 6:
        W_Comm += InputTotal
        print("User ", user,"-",demo, "-",Setup, " ,Input Total of ",InputTotal, "----- End State is at, " ,EndPos)
    if Setup == 1 or Setup == 5 or Setup == 3:
        Wo_Comm += InputTotal
        print("User ", user ,"-",demo,"-",Setup, " ,Input Total of ",InputTotal, "----- End State is at, " ,EndPos)

    #print()
if Wo_Comm > W_Comm:
    print("Success W_Comm required ", ((Wo_Comm/W_Comm)-1 )*100, " percent less inputs" )

#db = {'TotalTime':timed, 'SA_time':SA_time,'State':StateList,'UserAction':UserActionList,'AutoAction':AutoActionList,'InputList':InputList}
# dbfile = open(self.filename,'ab')
# pickle.dump(db,dbfile)
# dbfile.close()


# data1 = data2array("error1.pkl")
# data2 = data2array("error2.pkl")
# data3 = data2array("error3.pkl")

# error1 = get_error(data1)
# error2 = get_error(data2)
# error3 = get_error(data3)

# error = error1 + error2 + error3
# error = np.array(error)
# np.savetxt("error.csv", error, delimiter=",")

# # confirm all data is here
# print(np.shape(error))

# # get metrics
# mean = np.mean(error, axis=0)
# sem = np.std(error, axis=0) / np.sqrt(30)

# # plot result
# x = range(8)
# plt.bar(x, mean)
# plt.errorbar(x, mean, sem)
# plt.show()


# # regret processing
# data1 = pickle.load(open("regret1.pkl", "rb"))
# regret = data1
# np.savetxt("regret.csv", regret, delimiter=",")

# # confirm all data is here
# print(np.shape(regret))

# # get metrics
# mean = np.mean(regret, axis=0)
# sem = np.std(regret, axis=0) / np.sqrt(30)

# # plot result
# x = range(4)
# plt.bar(x, mean)
# plt.errorbar(x, mean, sem)
# plt.show()
