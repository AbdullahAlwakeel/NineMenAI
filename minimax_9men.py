from nine_men_morris import *
import random
import math
from numba import jit, njit
import numpy as np
from numba.typed import List

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
def SortMoves(State, Player, Moves):
    if len(Moves) <= 2:
        return Moves
    Evals = []
    for M in Moves:
        Evals.append(Evaluation(PlayMove(State, Player, M)))
    
    Sorted_Evals = sorted(Evals)
    Sorted_Evals.reverse()
    N_Moves = []
    for i in range(len(Moves)):
        m_ind = Evals.index(Sorted_Evals[i])
        N_Moves.append(Moves[m_ind])
    return N_Moves

@njit
def GetAllMoves(State, Player):
    if State.GetStonesInHand(Player) > 0: #player is placing down new stones
        FormedMill = False
        BlockedMill = False
        posss = [[np.int64(x)] for x in range(0)]
        for x in range(len(State.Board)):
                if State.Board[x] == 0:
                    if SemiMillAtPos(State, x, Player):
                        if not FormedMill:
                            posss.clear() #clear all those moves that don't get a mill
                            FormedMill = True
                        OppStones = GetPlayerStones(State, -Player)
                        for OS in OppStones:
                            posss.append( [NewPiece, x, OS])
                    elif SemiMillAtPos(State, x, -Player): #other player has a semi-mill, block it!!!
                        if not FormedMill: #though if there is a mill formed, priorotize that
                            if not BlockedMill:
                                posss.clear() #clear all moves that don't prevent opp from making a mill
                                BlockedMill = True
                            posss.append( [NewPiece, x])
                    else:
                        if (not FormedMill) and (not BlockedMill): #if the move aint get a mill and u found a mill move, then ignore it
                            posss.append( [NewPiece, x ])
        SortMoves(State, Player, posss)
        return posss
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
        SortMoves(State, Player, PossibleMoves)
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
        SortMoves(State, Player, PossibleMoves)
        return PossibleMoves

@njit
def Evaluation(State):
    if State.GameOver == True:
        if State.StonesInGame1 < 3:
            return -10000 #1st lost
        elif State.StonesInGame2 < 3:
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
        if State.GetStonesInHand(-1) > 0:
            return 1000*(StoneDifference)
        elif (State.GetStonesInGame(1)+State.GetStonesInHand(1) == 3) or (State.GetStonesInGame(-1)+State.GetStonesInHand(-1) == 3):
            return 700*(StoneDifference) + 200*(SemiMills1-SemiMills2)
        else:
            return 1000*(StoneDifference)+100*(NumberTrappedStones2-NumberTrappedStones1)+50*(NumberMoves1-NumberMoves2)
        
                


@njit
def GetBestMove(State, Player, Depth = 10): #top node function, returns evaluation + best move
    if (State.Board == np.zeros(24)).all(): #only consider a few starting moves since the rest are reflections (4 moves)
        Moves = [[-1, 0], [-1, 1], [-1, 3], [-1, 4]]
    else:
        Moves = GetAllMoves(State, Player)

    T_Calls = np.array([0])
    States_Table = [State_Class() for x in range(0)]
    ThisScore = 0
    M_ind = 0
    print(len(Moves), " possible moves at the beginning to be explored.")
    print(Moves)
    Alpha = -100000
    Beta = 100000
    if len(Moves) == 1:
        return Evaluation(State), Moves[0]
    if Player == 1:
        ThisScore = -99999
        for M in Moves:
            N_S = PlayMove(State, Player, M)
            eval_ = Minimax(N_S, -Player, Depth-1, Alpha=Alpha, Beta=Beta, T_Calls=T_Calls, States_Table=States_Table)
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
            eval_ = Minimax(N_S, -Player, Depth-1, Alpha=Alpha, Beta=Beta, T_Calls=T_Calls, States_Table=States_Table)
            if ThisScore > eval_:
                ThisScore = eval_
                M_ind = Moves.index(M)
            Beta = min(Beta, eval_)
            if Beta <= Alpha:
                print("pruned! top node")
                break
        
    return ThisScore, Moves[M_ind]


@njit()
def Minimax(State, Player, Depth = 10,Alpha=-99999,Beta=99999, T_Calls=np.array([0]), States_Table=[State_Class() for x in range(0)]):
    #global T_Calls, Calls
    T_Calls[0] += 1
    for s in States_Table:
        if SameState(s, State):
            return Evaluation(State) #prevent looping thru same states
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
        return Evaluation(State)
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
            eval_ = Minimax(N_S, -Player, Depth-1, Alpha=Alpha, Beta=Beta,T_Calls=T_Calls, States_Table=States_Table)
            if ThisScore < eval_:
                ThisScore = eval_
            Alpha = max(Alpha, eval_)
            if Beta <= Alpha:
                break
    else:
        ThisScore = 99999
        for M in Moves:
            N_S = PlayMove(State, Player, M)
            eval_ = Minimax(N_S, -Player, Depth-1, Alpha=Alpha, Beta=Beta,T_Calls=T_Calls, States_Table=States_Table)
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

CurrState__ = State_Class()
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

TotalMoves = 0


LoseInX = False
Mate_In_Depth = -1

while CurrState__.GameOver == False:
    if LoseInX:
        print("Losing in X moves.")
    elif Mate_In_Depth > -1:
        Depth = Mate_In_Depth-1
        print("Win in less than ", Depth, " moves")
        Mate_In_Depth -= 1
    elif CurrState__.StonesInHand2 > 4:
        Depth__ = 7
    else:
        Depth__ = min(6 + math.floor(TotalMoves/8), 9)
    print("Current Depth: ", Depth__)
    print("evaluation function: ", Evaluation(CurrState__))
    print("Total moves made: ",TotalMoves)
    if PlayerStart == False:
        Eval, Move = GetBestMove(CurrState__, 1, Depth=Depth__)
        CurrState__ = PlayMove(CurrState__, 1, Move)
        print("eval = "+str(Eval))
        if Eval == -10000:
            LoseInX = True
        elif Eval == 10000:
            Mate_In_Depth = Depth__
        print("move = ", Move)
    printState(CurrState__)
    print("Stones in Hands: 1: ",CurrState__.StonesInHand1, ", 2: ", CurrState__.StonesInHand2)
    print("Stones in Game: 1: ", CurrState__.StonesInGame1, ", 2: ", CurrState__.StonesInGame2)
    print("     %s          %s           %s\n          %s     %s     %s      \n             %s  %s  %s\n     %s   %s  %s    %s %s    %s\n             %s %s %s\n         %s     %s    %s    \n    %s          %s          %s\n" % tuple(range(24)))
    P_M = None
    GotInput = False
    while not GotInput:
        try:
            P_M = GetPInput()
            if PlayerStart:
                CurrState__ = PlayMove(CurrState__, 1, P_M)
            else:
                CurrState__ = PlayMove(CurrState__, -1, P_M) 
            GotInput = True     
        except:
            pass
    if PlayerStart == True:
        Eval, Move = GetBestMove(CurrState__, -1, Depth=Depth__)
        if Eval == -10000:
            Mate_In_Depth = Depth__
        elif Eval == 10000:
            LoseInX = True
        CurrState__ = PlayMove(CurrState__, -1, Move)
        print("eval = "+str(Eval))
        print("move = ", Move)
    TotalMoves += 2


print("Game over!")





        