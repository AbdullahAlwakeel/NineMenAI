#nine men solver, very optimized
import numpy as np
import numba
from numba.types import int8, int64
from numba.typed import List
import time

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
            [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27], #none
            [21,9,0,18,10,3,15,11,6,22,19,16,7,4,1,17,12,8,20,13,5,23,14,2, 24, 25, 26, 27], #90 degrees cclk
            [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,8,7,6,5,4,3,2,1,0, 24, 25, 26, 27], #180
            [2,14,23,5,13,20,8,12,17,1,4,7,16,19,22,6,11,15,3,10,18,0,9,21, 24, 25, 26, 27], #270 (90 clk)
        ], dtype="int8")
ref_sym = np.array([2,1,0,5,4,3,8,7,6,14,13,12,11,10,9,17,16,15,20,19,18,23,22,21, 24, 25, 26, 27], dtype="int8")
swap_sym = np.array([6,7,8,3,4,5,0,1,2,11,10,9,14,13,12,21,22,23,18,19,20,15,16,17, 24, 25, 26, 27], dtype="int8")

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
def Search(S, Player, Depth, PrevStates, T_Calls, Alpha, Beta): #simple minimax search with symmetries, repetition rules
    N_PS = [np.zeros_like(S)]
    T_Calls[0] += 1
    if T_Calls[0] % 100 == 0:
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
        return (S[24]+S[26])-(S[25]+S[27]) #NOTE: bad, replace with evaluation function
    N_PS.append(S)
    NewStates = [S]
    Values = [0]
    Values.remove(0)
    Moves = GetAllMoves(S, Player)
    if len(Moves) == 0:
        return -100 * Player
    for m in Moves:
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
            val = Search(NewState, -Player, Depth=Depth-1, PrevStates=N_PS, T_Calls=T_Calls, Alpha=Alpha, Beta=Beta) #recursion here
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
        return max(Values)
    else:
        return min(Values)

P_S = []
T_C = []

def GetBestMove(State, Player, Depth): #gonna use threading n stuff
    global T_C, P_S
    if len(P_S) == 0:
        P_S = [np.zeros(29, dtype="int8")]
    T_C = np.array([0], dtype="uint64")
    Moves = GetAllMoves(State, Player)
    N_States = []
    Vals = []
    acc_moves = []
    for m in Moves:
        new_state = PlayMove(State,Player,m)
        if len(N_States) == 0:
            N_States.append(new_state)
            acc_moves.append(m)
            #def Search(S, Player, Depth, PrevStates, T_Calls, Alpha, Beta):
            print("-----------------------------")
            print("Searching move ", str(len(N_States)-1) ,":")
            Vals.append(Search(new_state, -Player, Depth, P_S, T_C, -9999, 9999))
        else: #check for symmetries
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
                print("-----------------------------")
                print("Searching move ", str(len(N_States)-1) ,":")
                Vals.append(Search(new_state, -Player, Depth, P_S, T_C, -9999, 9999))
    if Player == 1:
        best_move = max(Vals)
        return Vals.index(best_move), acc_moves[Vals.index(best_move)], Vals, acc_moves
    else:
        best_move = min(Vals)
        return Vals.index(best_move), acc_moves[best_move], Vals, acc_moves


Start = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,9], dtype="int8") #start position, both players have 9 stones in hand
#Start = np.array([1,1,1,1,1,1,-1,-1,-1,0,1,1,0,0,-1,0,-1,0,-1,0,-1,0,-1,-1,9,9,0,0], dtype="int8") 
milli = time.time()
eval, move, vals, moves = GetBestMove(Start, 1, 6)
milli = time.time() - milli
print("number of calls: ",int(T_C[0]))
print("evaluation: ", eval)
print("best move: ", move)
print("time taken: ", milli, "s")
for m in range(len(moves)):
    print("move ",moves[m]," has evaluation ",vals[m])
print("nodes per second: ", int(T_C[0] / milli))