from nine_men_morris import *
import random
import math
from numba import jit, njit
import numpy as np
from numba.typed import List
from NeuralNet import NeuralNet

#@jit
def SortMoves(State, Player, Moves):
    if len(Moves) <= 2:
        return Moves
    Evals = []
    for M in Moves:
        Evals.append(Evaluation(PlayMove(State, Player, M), ))
    
    Sorted_Evals = sorted(Evals)
    Sorted_Evals.reverse()
    N_Moves = []
    for i in range(len(Moves)):
        m_ind = Evals.index(Sorted_Evals[i])
        N_Moves.append(Moves[m_ind])
    return N_Moves

#@jit
def GetAllMoves(State, Player):
    if State.GetStonesInHand(Player) > 0: #player is placing down new stones
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
                    elif SemiMillAtPos(State, x, -Player): #other player has a semi-mill, block it!!!
                        if not FormedMill: #though if there is a mill formed, priorotize that
                            if not BlockedMill:
                                p.clear() #clear all moves that don't prevent opp from making a mill
                                BlockedMill = True
                            p.append( [NewPiece, x])
                    else:
                        if (not FormedMill) and (not BlockedMill): #if the move aint get a mill and u found a mill move, then ignore it
                            p.append( [NewPiece, x ])
        return p
    elif State.GetStonesInGame(Player) > 3: #player can slide
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
                    else:
                        OppCanMakeMill = False
                        AdjToHole = GetConnections(A)
                        if State.GetStonesInGame(-Player) == 3: #opp can jujst fly to the position
                            OppCanMakeMill = True
                        else:
                            for spac in AdjToHole:
                                if spac == -Player:
                                    OppCanMakeMill = True
                                    break
                        if SemiMillAtPos(State, A, -Player, S) and OppCanMakeMill: #check if the player can even make a full mill before deciding to block it
                            if not FormedMill:
                                if not BlockedMill:
                                    PossibleMoves.clear()
                                    BlockedMill = True
                                PossibleMoves.append([S, A])
                        else:
                            if (not FormedMill) and (not BlockedMill): #if the move aint get a mill and u found a mill move, then ignore it
                                PossibleMoves.append( [S, A])
        return PossibleMoves
    elif State.GetStonesInGame(Player) == 3: #player can fly
        FormedMill = False
        BlockedMill = False
        PossibleMoves = [[np.int64(x)] for x in range(0)]
        PStones = GetPlayerStones(State, Player)
        for S in PStones:
            Adj = [np.int64(x) for x in range(0)]
            for p in range(len(State.Board)): #get all free positions in the board
                if State.Board[p] == 0:
                    Adj.append(p)
            for A in Adj:
                if State.Board[A] == 0:
                    if SemiMillAtPos(State, A, Player, S):
                        if not FormedMill:
                            PossibleMoves.clear() #clear all those moves that don't get a mill
                            FormedMill = True
                        OppStones = GetPlayerStones(State, -Player)
                        for OS in OppStones:
                            PossibleMoves.append( [S, A, OS])
                    else:
                        OppCanMakeMill = False
                        AdjToHole = GetConnections(A)
                        if State.GetStonesInGame(-Player) == 3: #opp can jujst fly to the position
                            OppCanMakeMill = True
                        else:
                            for spac in AdjToHole:
                                if spac == -Player:
                                    OppCanMakeMill = True
                                    break
                        if SemiMillAtPos(State, A, -Player, S) and OppCanMakeMill: #check if the player can even make a full mill before deciding to block it
                            if not FormedMill:
                                if not BlockedMill:
                                    PossibleMoves.clear()
                                    BlockedMill = True
                                PossibleMoves.append([S, A])
                        else:
                            if (not FormedMill) and (not BlockedMill): #if the move aint get a mill and u found a mill move, then ignore it
                                PossibleMoves.append( [S, A])
        return PossibleMoves

#@jit
def Evaluation(State, currPlayer):
    if State.GameOver == True:
        if State.StonesInGame1 < 3:
            return -10000 #1st lost
        elif State.StonesInGame2 < 3:
            return 10000 #1st won
    else:
        return currPlayer.Update(State.Board)
        
                


#@jit
def GetBestMove(State, Player, currPlayer, Depth = 10): #top node function, returns evaluation + best move
    if (State.Board == np.zeros(24)).all(): #only consider a few starting moves since the rest are reflections (4 moves)
        Moves = [[-1, 0], [-1, 1], [-1, 3], [-1, 4]]
    else:
        Moves = GetAllMoves(State, Player)

    T_Calls = np.array([0])
    States_Table = [State_Class() for x in range(0)]
    ThisScore = 0
    M_ind = 0
    #print(len(Moves), " possible moves at the beginning to be explored.")
    #print(Moves)
    Alpha = -100000
    Beta = 100000
    if len(Moves) == 1:
        return Evaluation(State, currPlayer), Moves[0]
    elif len(Moves) == 0:
        return -10000*Player, []
    if Player == 1:
        ThisScore = -99999
        for M in Moves:
            N_S = PlayMove(State, Player, M)
            eval_ = Minimax(N_S, -Player, currPlayer, Depth-1, Alpha=Alpha, Beta=Beta, T_Calls=T_Calls, States_Table=States_Table)
            if ThisScore < eval_:
                M_ind = Moves.index(M)
                ThisScore = eval_
            Alpha = max(Alpha, eval_)
            if Beta <= Alpha:
                print("pruned! top node")
                break
    else:
        ThisScore = 99999
        for M in Moves:
            N_S = PlayMove(State, Player, M)
            eval_ = Minimax(N_S, -Player, currPlayer, Depth-1, Alpha=Alpha, Beta=Beta, T_Calls=T_Calls, States_Table=States_Table)
            if ThisScore > eval_:
                ThisScore = eval_
                M_ind = Moves.index(M)
            Beta = min(Beta, eval_)
            if Beta <= Alpha:
                print("pruned! top node")
                break
        
    return ThisScore, Moves[M_ind]


#@jit()
def Minimax(State, Player, currPlayer, Depth = 10,Alpha=-99999,Beta=99999, T_Calls=np.array([0]), States_Table=[State_Class() for x in range(0)]):
    #global T_Calls, Calls
    T_Calls[0] += 1
    for s in States_Table:
        if SameState(s, State):
            return Evaluation(State, currPlayer) #prevent looping thru same states
    States_Table.append(State)
    if (State.GetStonesInHand(1)+State.GetStonesInGame(1) == 3) and (State.GetStonesInHand(-1)+State.GetStonesInGame(-1) == 3):
        Depth = min(2,Depth) #search 2 more moves then terminate, since position is likely a draw
    #while len(Calls) <= Depth:
    #    Calls.append(0)
    #Calls[Depth] += 1
    #    print(Calls)
    if T_Calls[0] % 5000 == 0:
        print(T_Calls)
    if Depth == 0 or State.GameOver==True:
        return Evaluation(State, currPlayer)
    Moves = GetAllMoves(State, Player)
    if len(Moves) == 0: #this an end state, with no legal moves
        return -10000*Player
    if Depth >= 8:
        print(Depth,": ",Player,": ",Moves)
    ThisScore = 0
    if Player == 1:
        ThisScore = -99999
        for M in Moves:
            N_S = PlayMove(State, Player, M)
            eval_ = Minimax(N_S, -Player, currPlayer, Depth-1, Alpha=Alpha, Beta=Beta,T_Calls=T_Calls, States_Table=States_Table)
            if ThisScore < eval_:
                ThisScore = eval_
            Alpha = max(Alpha, eval_)
            if Beta <= Alpha:
                break
    else:
        ThisScore = 99999
        for M in Moves:
            N_S = PlayMove(State, Player, M)
            eval_ = Minimax(N_S, -Player, currPlayer, Depth-1, Alpha=Alpha, Beta=Beta,T_Calls=T_Calls, States_Table=States_Table)
            if ThisScore > eval_:
                ThisScore = eval_
            Beta = min(Beta, eval_)
            if Beta <= Alpha:
                break
    return ThisScore

#move notation:
#     00         01          02
#          03    04    05      
#             06 07 08
#     09   10 11    12 13    14
#             15 16 17
#          18    19    20    
#     21         22          23

def printState(State):
    print("----------------------------------------")
    b = []
    for i in range(len(State.Board)):
        if State.Board[i] == -1:
            b.append("##")
        elif State.Board[i] == 0:
            b.append("..")
        else:
            b.append("OO")
    print("     %s         %s          %s\n          %s    %s    %s      \n             %s %s %s\n     %s   %s %s    %s %s    %s\n             %s %s %s\n          %s    %s    %s    \n     %s         %s          %s\n" % tuple(b))
    print("----------------------------------------------")


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


#CurrState__.Board = np.array([1,0,1,0,-1,0,1,1,-1,0,-1,1,0,0,0,1,-1,0,0,0,-1,1,-1,0], dtype=np.int8)
#CurrState__.StonesInGame1 = 7
#CurrState__.StonesInGame2 = 6
#CurrState__.StonesInHand1 = 0
#CurrState__.StonesInHand2 = 0

PlayerStart = False

Depth__ = 6

def GetPInput():
    s = int(input("enter s: "))
    d = int(input("enter d: "))
    t = int(input("enter t: "))
    P_M = [s, d, t]
    return P_M


PopulationSize = 100
Population = []
for i in range(PopulationSize):
    Population.append(NeuralNet(24, 1, 30))

import json
LS = []
with open("lastgame.txt", "r") as f:
    LS = json.load(f)

St = State_Class()
CP = 1
for Move in LS:
    printState(St)
    St = PlayMove(St, CP, Move)
    CP = -CP

Depth = 1

P1Turn = True
TotalSteps = 0

LastGamePlayed = []

for Iterations in range(10000):
    LastGamePlayed = []
    StepsHere = 0
    print("Starting Iteration ", Iterations)
    P1_Ind = random.randint(0, PopulationSize-1)
    P1 = Population[P1_Ind]
    P2_Ind = P1_Ind
    while P2_Ind == P1_Ind:
        P2_Ind = random.randint(0, PopulationSize-1)
    P2 = Population[P2_Ind]
    
    print("p1: ", P1_Ind, ", p2: ", P2_Ind)
    CurrState__ = State_Class()

    Draw = False

    Eval__ = 0

    while CurrState__.GameOver == False:
        if P1Turn:
            Eva, Move = GetBestMove(CurrState__, 1, P1, Depth=Depth)
            if Eva == 10000 or Eva == -10000:
                Eval__ = Eva
                break
            LastGamePlayed.append(Move)
            CurrState__ = PlayMove(CurrState__, 1, Move)
            P1Turn = False
        else:
            Eva, Move = GetBestMove(CurrState__, -1, P2, Depth=Depth)
            if Eva == 10000 or Eva == -10000:
                Eval__ = Eva
                break
            LastGamePlayed.append(Move)
            CurrState__ = PlayMove(CurrState__, -1, Move)
            P1Turn = True
        TotalSteps += 1
        StepsHere += 1
        if StepsHere > 100:
            Draw = True
            break
        if TotalSteps % 100 == 0:
            print(TotalSteps)
        if TotalSteps % 10000 == 0:
            printState(CurrState__)
        
    
    if (not Draw):
        if Eval__ > 0: #P1 wins
            print("P1 won!")
            P2.Copy(P1)
            P2.Mutate()
        else:
            print("P2 won!")
            P1.Copy(P2)
            P1.Mutate()
    else:
        if CurrState__.StonesInGame1 > CurrState__.StonesInGame2:
            print("P1 won!")
            P2.Copy(P1)
            P2.Mutate()
        elif CurrState__.StonesInGame1 < CurrState__.StonesInGame2:
            print("P2 won!")
            P1.Copy(P2)
            P1.Mutate()
        else:
            print("choosing random winner...")
            if random.random() <= 0.5:
                print("P1 won!")
                P2.Copy(P1)
                P2.Mutate()
            else:
                print("P2 won!")
                P1.Copy(P2)
                P1.Mutate()

        #choose randomly

print("Game over!")
with open("lastgame.txt", "w") as f:
    json.dump(LastGamePlayed, f)






        