from six_men_morris import *
import random
import math
from numba import jit, njit
import numpy as np
#from numba.typed import List

@njit
def Shuffle(L):
    if len(L) <= 2:
        return L
    N_L = []
    indx = []
    for i in range(len(L)):
        indx.append(i)
    for i in range(len(L)):
        a = random.randint(0, len(L)-1)
        b = a
        while b == a:
            b = random.randint(0, len(L)-1)
        tmp = indx[a]
        indx[a] = indx[b]
        indx[b] = tmp
    for i in indx:
        N_L.append(L[i])
    return N_L

@njit
def GetAllMoves(State, Player):
    if State.GetStonesInHand(Player) > 0:
        FormedMill = False
        BlockedMill = False
        p = [[np.int64(x)] for x in range(0)]
        for x in range(len(State.Board)):
                if State.Board[x] == 0:
                    if SemiMillAtPos(State, x, Player):
                        if not FormedMill:
                            p.clear() #clear all those moves that don't get a mill
                            FormedMill = True
                        OppStones = GetPlayerStones(State, -Player)
                        for OS in OppStones:
                            p.append( [NewPiece, x, OS])
                    elif SemiMillAtPos(State, x, -Player): #block that mill!
                        if not FormedMill: #though if there is a mill formed, priorotize that
                            if not BlockedMill:
                                p.clear() #clear all moves that don't prevent opp from making a mill
                                BlockedMill = True
                            p.append( [NewPiece, x])
                    else:
                        if (not FormedMill) and (not BlockedMill): #if the move aint get a mill and u found a mill move, then ignore it
                            p.append( [NewPiece, x ])
        p = Shuffle(p)
        return p
    else:
        FormedMill = False
        BlockedMill = False
        PossibleMoves = [[np.int64(x)] for x in range(0)]
        PStones = GetPlayerStones(State, Player)
        for S in PStones:
            Adj = GetConnections(S)
            for A in Adj:
                if State.Board[A] == 0:
                    if SemiMillAtPos(State, A, Player, S):
                        if not FormedMill:
                            PossibleMoves.clear() #clear all those moves that don't get a mill
                            FormedMill = True
                        OppStones = GetPlayerStones(State, -Player)
                        for OS in OppStones:
                            PossibleMoves.append( [S, A, OS])
                    elif SemiMillAtPos(State, A, -Player, S):
                        if not FormedMill:
                            if not BlockedMill:
                                PossibleMoves.clear()
                                BlockedMill = True
                            PossibleMoves.append([S, A])
                    else:
                        if (not FormedMill) and (not BlockedMill): #if the move aint get a mill and u found a mill move, then ignore it
                            PossibleMoves.append( [S, A])
        PossibleMoves = Shuffle(PossibleMoves)
        return PossibleMoves

@njit
def Evaluation(State):
    if State.GameOver == True:
        if State.StonesInGame1 < 3:
            return -10000 #1st lost
        else:
            return 10000 #1st won
    else:
        #get number & location of stones for each player
        PlayerStones1 = GetPlayerStones(State, 1)
        PlayerStones2 = GetPlayerStones(State, -1)
        StoneDifference = (State.GetStonesInHand(1)+State.GetStonesInGame(1))-(State.GetStonesInHand(-1)+State.GetStonesInGame(-1))

        #get number of possible moves, and also get number of trapped stones
        NumberTrappedStones1 = 0
        NumberTrappedStones2 = 0
        NumberMoves1 = 0
        NumberMoves2 = 0
        SemiMills1 = 0
        SemiMills2 = 0
        Mills1 = 0
        Mills2 = 0
        for St in PlayerStones1:
            if SemiMillAtPos(State, St, 1):
                SemiMills1 += 1
            if MillAtPos(State, St):
                Mills1 += 1
            Adj = GetConnections(St)
            HasMoves = False
            for p in Adj:
                if not (State.Board[p] == 0):
                    HasMoves = True
                    NumberMoves1 += 1
            if HasMoves == False:
                NumberTrappedStones1 += 1
        #-----------------------
        for St in PlayerStones2:
            if SemiMillAtPos(State, St, -1):
                SemiMills2 += 1
            if MillAtPos(State, St):
                Mills2 += 1
            Adj = GetConnections(St)
            HasMoves = False
            for p in Adj:
                if not (State.Board[p] == 0):
                    HasMoves = True
                    NumberMoves2 += 1
            if HasMoves == False:
                NumberTrappedStones2 += 1
        #return 20*(StoneDifference)+10*(NumberTrappedStones2-NumberTrappedStones1)+(NumberMoves1-NumberMoves2)+50*(Mills1-Mills2)+10*(SemiMills1-SemiMills2) #test
        if State.GetStonesInHand(-1) == 0:
            return 2000*(StoneDifference)+100*(NumberTrappedStones2-NumberTrappedStones1)+150*(NumberMoves1-NumberMoves2)
        else:
            return 1000*(StoneDifference)
        
                

                
Calls = [0]
T_Calls = 0

@njit
def GetBestMove(State, Player, Depth = 10):
    Moves = GetAllMoves(State, Player)
    T_Calls = np.array([0])
    ThisScore = 0
    M_ind = 0
    Alpha=-99999
    Beta=99999
    if Player == 1:
        ThisScore = -99999
        for M in Moves:
            N_S = PlayMove(State, Player, M)
            eval_ = Minimax(N_S, -Player, Depth-1, Alpha=Alpha, Beta=Beta, T_Calls=T_Calls)
            if ThisScore < eval_:
                M_ind = Moves.index(M)
                ThisScore = eval_
            Alpha = max(Alpha, eval_)
            if Beta <= Alpha:
                break
    else:
        ThisScore = 99999
        for M in Moves:
            N_S = PlayMove(State, Player, M)
            eval_ = Minimax(N_S, -Player, Depth-1, Alpha=Alpha, Beta=Beta, T_Calls=T_Calls)
            if ThisScore > eval_:
                ThisScore = eval_
                M_ind = Moves.index(M)
            Beta = min(Beta, eval_)
            if Beta <= Alpha:
                break
        
    return ThisScore, Moves[M_ind]


@njit()
def Minimax(State, Player, Depth = 10,Alpha=-99999,Beta=99999, T_Calls=np.array([0])):
    #global T_Calls, Calls
    T_Calls[0] += 1
    #while len(Calls) <= Depth:
    #    Calls.append(0)
    #Calls[Depth] += 1
    #    print(Calls)
    if T_Calls[0] % 1000 == 0:
        print(T_Calls)
    if Depth == 0 or State.GameOver==True:
        return Evaluation(State)
    Moves = GetAllMoves(State, Player)
    if len(Moves) == 0: #this an end state
        return -10000*Player

    ThisScore = 0
    M_ind = 0
    if Player == 1:
        ThisScore = -99999
        for M in Moves:
            N_S = PlayMove(State, Player, M)
            eval_ = Minimax(N_S, -Player, Depth-1, Alpha=Alpha, Beta=Beta,T_Calls=T_Calls)
            if ThisScore < eval_:
                M_ind = Moves.index(M)
                ThisScore = eval_
            Alpha = max(Alpha, eval_)
            if Beta <= Alpha:
                break
    else:
        ThisScore = 99999
        for M in Moves:
            N_S = PlayMove(State, Player, M)
            eval_ = Minimax(N_S, -Player, Depth-1, Alpha=Alpha, Beta=Beta,T_Calls=T_Calls)
            if ThisScore > eval_:
                ThisScore = eval_
                M_ind = Moves.index(M)
            Beta = min(Beta, eval_)
            if Beta <= Alpha:
                break
    return ThisScore

#move notation:
#     0          1          2
#           3    4    5      
#     6     7         8     9
#          10   11    12    
#     13        14          15

def printState(State):
    print("---------")
    b = []
    for i in range(len(State.Board)):
        if State.Board[i] == -1:
            b.append(2)
        elif State.Board[i] == 0:
            b.append(".")
        else:
            b.append(State.Board[i])
    print("%s   %s   %s\n  %s %s %s\n%s %s   %s %s\n  %s %s %s\n%s   %s   %s" % tuple(b))
    print("---------")


#TestState = {
#    "Board":[[1,1,1],
#    [0,-1,0],
#    [-1,1,1,0],
#    [-1,1,0],
#    [0,0,-1]
#    ],
#    "StonesInHand":{1:0,-1:0}, #each player starts 6 stones in hand
#    "StonesInGame":{1:6,-1:4},
#    "GameOver":False
#}

CurrState__ = State_Class()
"""CurrState__.Board = np.array([1,0,0,
0,0,0,
-1,-1,0,0,
0,0,0,
1,-1,1])
CurrState__.StonesInGame1 = 3
CurrState__.StonesInGame2 = 3
CurrState__.StonesInHand1 = 3
CurrState__.StonesInHand2 = 3
"""
PlayerStart = False

Depth__ = 6

def GetPInput():
    s = int(input("enter s: "))
    d = int(input("enter d: "))
    t = int(input("enter t: "))
    P_M = [s, d, t]
    return P_M

TotalMoves = 0

while CurrState__.GameOver == False:
    if CurrState__.StonesInHand2 > 4:
        Depth__ = 8
    else:
        Depth__ = min(7 + math.floor(TotalMoves/4), 11)
    print("Current Depth: ", Depth__)
    print("Total moves made: ",TotalMoves)
    if PlayerStart == False:
        Eval, Move = GetBestMove(CurrState__, 1, Depth=Depth__)
        CurrState__ = PlayMove(CurrState__, 1, Move)
        print("eval = "+str(Eval))
        print("move = ", Move)
    printState(CurrState__)
    print("0          1          2")
    print("      3    4    5")
    print("6     7         8     9")
    print("     10   11    12 "   )
    print("13        14          15")
    P_M = None
    GotInput = False
    while not GotInput:
        try:
            P_M = GetPInput()
            GotInput = True
        except:
            pass

    if PlayerStart == False:
        CurrState__ = PlayMove(CurrState__, -1, P_M)
    else:
        CurrState__ = PlayMove(CurrState__, 1, P_M)
        Eval, Move = GetBestMove(CurrState__, -1, Depth=Depth__)
        CurrState__ = PlayMove(CurrState__, -1, Move)
        print("eval = "+str(Eval))
        print("move = ", Move)
    TotalMoves += 2





        