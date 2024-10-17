#nine men solver, very optimized
import numpy as np
import numba
from numba.types import int8, int64
from numba.typed import List
from numba import cuda
import time
from NeuralNet import *
import random

@numba.njit
def GetConnections(p):
    Connections = [
    [1,9],
    [0,2,4],
    [1,14],
    [4,10],
    [1,3,5,7],
    [4,13],
    [7,11],
    [4,6,8],
    [7,12],
    [0,10,21],
    [9,3,11,18],
    [6,10,15],
    [8,13,17],
    [5,12,14,20],
    [2,13,23],
    [11,16],
    [15,17,19],
    [12,16],
    [10,19],
    [16,18,20,22],
    [13,19],
    [9,22],
    [19,21,23],
    [14,22]
    ]
    return Connections[p]

@numba.njit
def GetMills():
    Mills = [
    [0,1,2],
    [3,4,5],
    [6,7,8],
    [15,16,17],
    [18,19,20],
    [21,22,23],
    [9,10,11],
    [12,13,14],
    [0,9,21],
    [3,10,18],
    [6,11,15],
    [8,12,17],
    [5,13,20],
    [2,14,23],
    [1,4,7],
    [16,19,22]
    ]
    return Mills

@numba.njit
def IsAdjacent(p1, p2): #will return false if p1 == p2
    return (p2 in GetConnections(p1))


@numba.njit
def MillAtPos(State,p):
    Player = State[p]
    for Mill in GetMills():
        if p in Mill:
            MilHere = True
            for pos in Mill:
                if not (State[pos] == Player):
                    MilHere = False
                    break
            if MilHere:
                return True
    return False

@numba.njit
def PlayMove(S, Player, Move): #Player = 1 (starting player), -1 (2nd player)
    State = np.copy(S)
    SInGame = 0
    SInHand = 0
    if Player == 1:
        SInGame = 24
        SInHand = 26
    else:
        SInGame = 25
        SInHand = 27
    
    if S[SInHand] > 0:
        State[Move[1]] = Player
        State[SInGame] += 1
        State[SInHand] -= 1
    else:
        IsLegal = True
        rsn = ""
        if not ((State[Move[0]] == Player) and (State[Move[1]] == 0)):
            IsLegal = False
            rsn = "start pos not player or end pos not empty"
        elif State[SInGame] > 3: #dont check for adjacency when in flying mode, any stone can go anywhere
            if not Move[1] in GetConnections(Move[0]):
                IsLegal = False
                rsn = "not adjacent"
        if not IsLegal:
            print("illegal move. "+rsn)
            print("state = ",S)
            print("player = ",Player)
            print("move = ",Move)
            raise Exception("illegal move. ")
        else:
            State[Move[0]] = 0
            State[Move[1]] = Player #do the move
    #check if a new mill formed
    NewMill = MillAtPos(State, Move[1])
    if NewMill: #remove one of opponent's piece
        State[Move[2]] = 0
        OpSInGame = 0
        if Player == 1:
            OpSInGame = 25
        else:
            OpSInGame = 24
        State[OpSInGame] -= 1
    State[28] = -State[28]
    return State

@numba.njit
def SemiMillAtPos(State,p, Player, Orig_p=-1): #2 out of 3 are full by same player
    for Mill in GetMills():
        if p in Mill:
            NumPlayer = 0
            for pos in Mill:
                if (State[pos] == Player):
                    if not (pos == Orig_p):
                        NumPlayer += 1
                elif (State[pos] == -Player):
                    NumPlayer = 0
                    break
            if NumPlayer == 2:
                return True
    return False

@numba.njit
def GetAllMoves(S, Player):
    Moves = []
    SInGame = 0
    SInHand = 0
    if Player == 1:
        SInGame = 24
        SInHand = 26
    else:
        SInGame = 25
        SInHand = 27
    if S[SInHand] > 0: #get all empty squares
        for x in range(24):
            if S[x] == 0:
                if SemiMillAtPos(S, x, Player):
                    for j in range(24):
                        if S[j] == -Player:
                            Moves.append([-1, x, j])
                else:
                    Moves.append([-1, x, -1])
    elif S[SInGame] > 3:
        for x in range(24):
            if S[x] == Player:
                for y in GetConnections(x):
                    if S[y] == 0:
                        if SemiMillAtPos(S, y, Player, x):
                            for j in range(24):
                                if S[j] == -Player:
                                    Moves.append([x, y, j])
                        else:
                            Moves.append([x, y, -1])
    else:
        for x in range(24):
            if S[x] == Player:
                for y in range(24):
                    if S[y] == 0:
                        if SemiMillAtPos(S, y, Player, x):
                            for j in range(24):
                                if S[j] == -Player:
                                    Moves.append([x, y, j])
                        else:
                            Moves.append([x, y, -1])
    return Moves

@numba.njit     
def ApplySym(a, s):
    r = np.zeros(29, dtype=int8)
    for i in range(len(a)):
        r[i] = a[s[i]]
    return r

@numba.njit
def ApplySymMove(m, s):
    r = []
    for i in range(len(m)):
        if m[i] == -1:
            r.append(-1)
        else:
            r.append(s[m[i]])
    return r

SymmetriesArrs = []
rot_sym = np.array([
            [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27, 28], #none
            [21,9,0,18,10,3,15,11,6,22,19,16,7,4,1,17,12,8,20,13,5,23,14,2, 24, 25, 26, 27, 28], #90 degrees cclk
            [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,8,7,6,5,4,3,2,1,0, 24, 25, 26, 27, 28], #180
            [2,14,23,5,13,20,8,12,17,1,4,7,16,19,22,6,11,15,3,10,18,0,9,21, 24, 25, 26, 27, 28], #270 (90 clk)
        ], dtype="int8")
ref_sym = np.array([2,1,0,5,4,3,8,7,6,14,13,12,11,10,9,17,16,15,20,19,18,23,22,21, 24, 25, 26, 27, 28], dtype="int8")
swap_sym = np.array([6,7,8,3,4,5,0,1,2,11,10,9,14,13,12,21,22,23,18,19,20,15,16,17, 24, 25, 26, 27, 28], dtype="int8")

SymmetriesArrs.append(rot_sym[1])
SymmetriesArrs.append(rot_sym[2])
SymmetriesArrs.append(rot_sym[3])
SymmetriesArrs.append(ref_sym)
SymmetriesArrs.append(swap_sym)

SymmetriesArrs.append(ApplySym(rot_sym[0], ref_sym))
SymmetriesArrs.append(ApplySym(rot_sym[0], swap_sym))
SymmetriesArrs.append(ApplySym(ApplySym(rot_sym[0], ref_sym), swap_sym))

SymmetriesArrs.append(ApplySym(rot_sym[1], ref_sym))
SymmetriesArrs.append(ApplySym(rot_sym[1], swap_sym))
SymmetriesArrs.append(ApplySym(ApplySym(rot_sym[1], ref_sym), swap_sym))

SymmetriesArrs.append(ApplySym(rot_sym[2], ref_sym))
SymmetriesArrs.append(ApplySym(rot_sym[2], swap_sym))
SymmetriesArrs.append(ApplySym(ApplySym(rot_sym[2], ref_sym), swap_sym))

SymmetriesArrs.append(ApplySym(rot_sym[3], ref_sym))
SymmetriesArrs.append(ApplySym(rot_sym[3], swap_sym))
SymmetriesArrs.append(ApplySym(ApplySym(rot_sym[3], ref_sym), swap_sym))

SymmetriesArrs = np.array(SymmetriesArrs, dtype="int8")

@numba.njit
def PlayFullGame(S, Player, PrevStates, Net, temp=1.0): #plays a game randomly(almost) until a terminal position is reached
    if S[24]+S[26] < 3:
        PrevStates.append(S)
        return -1.0
    elif S[25]+S[27] < 3:
        PrevStates.append(S)
        return 1.0
    else: #check for draw
        for sT in range(len(PrevStates)):
            if (PrevStates[sT] == S).all():
                return 0.0 #draw
    PrevStates.append(S)
    Moves = GetAllMoves(S, Player)
    if len(Moves) == 0:
        return -1.0 * Player
    ChosenMove = [-1, -1, -1]
    if random.random() < temp: #choose randomly
        i = random.randint(0, len(Moves)-1)
        ChosenMove = Moves[i]
    else: #choose greedily according to neural net evaluation of each board
        Evals = []
        for m in Moves:
            n_st = PlayMove(S, Player, m)
            Evals.append(Net.Update(n_st.astype(np.float64))[0])
        if Player == 1:
            ChosenMove = Moves[Evals.index(max(Evals))]
        else:
            ChosenMove = Moves[Evals.index(min(Evals))]
    return PlayFullGame(PlayMove(S, Player, ChosenMove), -Player, PrevStates, Net, temp=(temp*0.94))

NumVisited = {} #number of times each state was visited
Outcomes = {} #average outcome for each state
def LearnGame(Net, NumEpisodes=100):
    Start = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,9,1], dtype="int8") #start
    numP2Wins = 0
    numP1Wins = 0
    global NumVisited, Outcomes
    for _episode_ in range(NumEpisodes):
        States_Visited = [np.zeros(29, dtype="int8")]
        OC = PlayFullGame(Start, 1, States_Visited, Net, temp=(1-(_episode_/NumEpisodes))) * ((_episode_+50)/(NumEpisodes+50))
        if OC >= 0.1:
            numP1Wins += 1
        elif OC <= -0.1:
            numP2Wins += 1
        States_Visited.pop(0)
        for st__ in States_Visited:
            st = st__.tostring()
            if st in Outcomes:
                Outcomes[st] = ((Outcomes[st]*NumVisited[st])+OC)/(NumVisited[st]+1)
                NumVisited[st] += 1
            else:
                Outcomes[st] = OC
                NumVisited[st] = 1
            for sym in range(SymmetriesArrs.shape[0]):
                new_st__ = ApplySym(st__, SymmetriesArrs[sym])
                new_st = new_st__.tostring()
                if new_st == st:
                    continue
                if new_st in Outcomes:
                    Outcomes[new_st] = ((Outcomes[new_st]*NumVisited[new_st])+OC)/(NumVisited[new_st]+1) #merge
                    NumVisited[new_st] += 1
                else:
                    Outcomes[new_st] = Outcomes[st]
                    NumVisited[new_st] = NumVisited[st]
        if _episode_ % 50 == 0:
            print("episode done ",_episode_," out of ",NumEpisodes)
    print("--------------")
    print("done, training now...")
    print("number of times player 1 won: ", numP1Wins)
    print("number of times player 2 won: ", numP2Wins)
    print("number of training samples: ", len(Outcomes))
    print("shuffling training set")
    n_err = 1
    avg_err = 0
    NumRepeats = 5
    trainingData = []
    avgOutcome = 0.0
    for st in Outcomes:
        state = np.fromstring(st, dtype="int8")
        avgOutcome += Outcomes[st]
        trainingData.append((state.astype(np.float64), np.array([Outcomes[st]])))
    avgOutcome = avgOutcome / len(Outcomes)
    print("avg outcome = ", avgOutcome)
    random.shuffle(trainingData)
    for i____ in range(NumRepeats):
        for Pair in trainingData:
            avg_err = ((avg_err*n_err) + Net.Learn(Pair[0], Pair[1])) / (n_err + 1)
            n_err += 1
            if n_err % 10000 == 0:
                print("average error: ", avg_err)
                print("number iterations: ", n_err, " out of ", len(Outcomes)*NumRepeats)
                print("state = ", Pair[0])
                print("outcome = ", Pair[1])
                print("net prediction = ", Net.Update(Pair[0])[0] )



@numba.njit
def SearchNet(S, Player, Depth, PrevStates, T_Calls, Alpha, Beta, Net): #simple minimax search with symmetries, repetition rules
    N_PS = [np.zeros_like(S)]
    T_Calls[0] += 1
    if T_Calls[0] % 10000 == 0:
        print(T_Calls[0])
    #first check if state equals any previous states, this is a draw
    #then check if there are less than 3 pieces with one of the players, also check if both have less than 3 pieces, then its a draw
    if S[24]+S[26] < 3:
        return -100
    elif S[25]+S[27] < 3:
        return 100
    else: #check for draw
        for sT in range(len(PrevStates)):
            if not (PrevStates[sT] == N_PS[0]).all():
                N_PS.append(PrevStates[sT])
            if (PrevStates[sT] == S).all():
                return 1e-4 #draw
        #TODO: also check for 3v3 draws
    if Depth == 0:
        return Net.Update(S.astype(np.float64))[0]
    N_PS.append(S)
    NewStates = [S]
    Values = [0]
    Values.remove(0)
    Moves = GetAllMoves(S, Player)
    if len(Moves) == 0:
        return -100 * Player
    
    #sort moves, using bubble sort and the network's evaluation
    sorted_moves = []
    for i in range(len(Moves)):
        target = 99999.0
        t_ind = -1
        if Player == -1:
            target = -99999.0
        for j in range(len(Moves)):
            St_ = PlayMove(S, Player, Moves[j])
            evaluation = Net.Update(St_.astype(np.float64))[0]
            if (Player == 1 and evaluation > target) or (Player == -1 and evaluation < target):
                target = evaluation
                t_ind = j
        sorted_moves.append(Moves[t_ind])
        Moves.remove(Moves[t_ind])


    for m in sorted_moves:
        NewState = PlayMove(S, Player, m)
        #check if position is symmetric to another position:
        IsSymmetric = False
        for s in range(SymmetriesArrs.shape[0]):
            for ns in NewStates:
                if (ns == ApplySym(NewState, SymmetriesArrs[s])).all():
                    IsSymmetric = True
                    break
            if IsSymmetric:
                break
        if IsSymmetric:
            continue
        else:
            NewStates.append(NewState)
            val = SearchNet(NewState, -Player, Depth=Depth-1, PrevStates=N_PS, T_Calls=T_Calls, Alpha=Alpha, Beta=Beta, Net=Net) #recursion here
            Values.append(val)
            if Player == 1:
                if Alpha < val:
                    Alpha = val
            else:
                if Beta > val:
                    Beta = val
            if Alpha >= Beta:
                break
    
    if Player == 1:
        if max(Values) > 90.0:
            return max(Values)-1.0 #lessen the value
        elif max(Values) < -90.0:
            return max(Values)+1.0 #increase the value
        else:
            return max(Values)
    else:
        if min(Values) < -90.0:
            return min(Values)+1.0 #lessen the value slightly for longer forced wins
        elif min(Values) > 90.0:
            return min(Values)-1.0 #increase the value slightly for longer forced losses (stretch it out)
        else:
            return min(Values)

P_S = []
T_C = []

def GetBestMove(State, Player, Depth, NNet): #gonna use threading n stuff
    global T_C, P_S
    if len(P_S) == 0:
        P_S = [np.zeros(29, dtype="int8")]
    T_C = np.array([0], dtype="uint64")
    Moves = GetAllMoves(State, Player)
    N_States = []
    Vals = []
    acc_moves = []
    Alpha = -99999
    Beta = 99999

    #sort moves (taken from SearchNet)
    sorted_moves = []
    for i in range(len(Moves)):
        target = 99999.0
        t_ind = -1
        if Player == -1:
            target = -99999.0
        for j in range(len(Moves)):
            St_ = PlayMove(State, Player, Moves[j])
            evaluation = NNet.Update(St_.astype(np.float64))[0]
            if (Player == 1 and evaluation > target) or (Player == -1 and evaluation < target):
                target = evaluation
                t_ind = j
        sorted_moves.append(Moves[t_ind])
        Moves.remove(Moves[t_ind])

    for m in sorted_moves:
        new_state = PlayMove(State,Player,m)
        #check for symmetries
        IsSymmetric = False
        for s in range(SymmetriesArrs.shape[0]):
            for ns in N_States:
                if (ns == ApplySym(new_state, SymmetriesArrs[s])).all():
                    IsSymmetric = True
                    break
            if IsSymmetric:
                break
        if IsSymmetric:
            continue
        else:
            N_States.append(new_state)
            acc_moves.append(m)
            #def Search(S, Player, Depth, PrevStates, T_Calls, Alpha, Beta):
            #print("-----------------------------")
            print("Searching move ", str(len(N_States)-1) ,":")
            Vals.append(SearchNet(new_state, -Player, Depth, P_S, T_C, Alpha, Beta, NNet))
            if Player == 1:
                Alpha = max(max(Vals), Alpha)
            else:
                Beta = min(min(Vals), Beta)
            if Alpha >= Beta:
                print("alpha-beta pruned!!!!")
                break
    if Player == 1:
        eval_ = max(Vals)
        return eval_, acc_moves[Vals.index(eval_)], Vals, acc_moves
    else:
        eval_ = min(Vals)
        return eval_, acc_moves[Vals.index(eval_)], Vals, acc_moves


def printState(State):
    print("----------------------------------------")
    b = []
    for i in range(24):
        if State[i] == -1:
            b.append("##")
        elif State[i] == 0:
            b.append("..")
        else:
            b.append("OO")
    print("     %s         %s          %s\n          %s    %s    %s      \n             %s %s %s\n     %s   %s %s    %s %s    %s\n             %s %s %s\n          %s    %s    %s    \n     %s         %s          %s\n" % tuple(b))
    print("stones in hand 1: ", State[26])
    print("stones in game 1: ", State[24])
    print("stones in hand 2: ", State[27])
    print("stones in game 2: ", State[25])
    print("----------------------------------------------")

def PlayAgainstPlayer(NetPlayer = -1):
    Depth = 5
    NNet = LoadNet("eval_net.net")
    CurrState = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,9,1], dtype="int8") #start position, both players have 9 stones in hand
    CurrPlayer = 1
    while True:
        printState(CurrState)
        Move = [-1, -1, -1]
        if CurrPlayer == NetPlayer:
            eval_, Move, v_, m__ = GetBestMove(CurrState, CurrPlayer, Depth, NNet)
            print("NNet plays ", Move)
            print("NNet evaluation: ", eval_)
        else:
            print("     %s          %s           %s\n          %s     %s     %s      \n             %s  %s  %s\n     %s   %s  %s    %s %s    %s\n             %s %s %s\n         %s     %s    %s    \n    %s          %s          %s\n" % tuple(range(24)))
            ValidMoves = GetAllMoves(CurrState, CurrPlayer)
            while not (Move in ValidMoves):
                try:
                    sor = int(input("source: "))
                    dst = int(input("dest: "))
                    tar = int(input("target: "))
                    Move = [sor, dst, tar]
                except:
                    print("no")
                    pass
        CurrState = PlayMove(CurrState, CurrPlayer, Move)
        CurrPlayer = -CurrPlayer


def BotAgainstBot(n1, n2):
    Depth = 5
    NNet1 = LoadNet(n1)
    NNet2 = LoadNet(n2)
    NetDict = {1:NNet1, -1:NNet2}
    CurrState = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,9,1], dtype="int8") #start position, both players have 9 stones in hand
    CurrPlayer = 1
    MoveList = []
    while True:
        printState(CurrState)
        Move = [-1, -1, -1]
        eval_, Move, v_, m__ = GetBestMove(CurrState, CurrPlayer, Depth, NetDict[CurrPlayer])
        print("NNet player ",CurrPlayer," is playing: ")
        print("NNet plays ", Move)
        print("NNet evaluation: ", eval_)
        CurrState = PlayMove(CurrState, CurrPlayer, Move)
        CurrPlayer = -CurrPlayer
        if eval_ > 90 or eval_ < -90:
            print("game is pretty much over.")
        MoveList.append(Move)
        if CurrState[24]+CurrState[26] < 3 or CurrState[25]+CurrState[27] < 3:
            break
    print("end of game! ")
    print("Move list = ", MoveList)
        


net_name = "eval_net3.net"
NNet = NeuralNet(29,100,1)
#NNet = LoadNet(net_name)
NNet.LearningRate = 0.1
Start = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,9,1], dtype="int8") #start position, both players have 9 stones in hand

BotAgainstBot("eval_net2.net", "eval_net.net")
#eval_, move_, v__, m__ = GetBestMove(Start, 1, 5, NNet)
#print("eval = ", eval_)
#print("move = ", move_)
#print("other vals: ", v__)
#print("other moves: ", m__)

"""
Iterations = 10
NumEpisodes = 10000
Exploration_Noise = 0.1
for i__ in range(Iterations):
    print("iteration number        ", i__)
    LearnGame(NNet, NumEpisodes)
    SaveNet(NNet, net_name)
    if i__ > 10:
        NNet.LearningRate = 0.002
    elif i__ > 3:
        NNet.LearningRate = 0.01


SaveNet(NNet, net_name)
"""