#6men game
import copy
import numba, numba.experimental
from numba import jit, njit
import array
import numpy as np
from numba import int32, int64, boolean, uint8, int8    # import the types

spec = [
    ('Board', int8[:]),               # a simple scalar field
    ('StonesInHand1', uint8),
    ('StonesInHand2', uint8),
    ("StonesInGame1", uint8),
    ("StonesInGame2", uint8),
    ("GameOver", boolean)
]
@numba.experimental.jitclass(spec)
class State_Class():
    def __init__(self):
        self.Board = np.zeros(16, dtype=np.int8)
        self.StonesInHand1 = 6
        self.StonesInHand2 = 6
        self.StonesInGame1 = 0
        self.StonesInGame2 = 0
        self.GameOver = False
        pass
    def GetStonesInHand(self, Player):
        if Player == 1:
            return self.StonesInHand1
        else:
            return self.StonesInHand2
    def SetStonesInHand(self, Player, Val):
        if Player == 1:
            self.StonesInHand1 = Val
        else:
            self.StonesInHand2 = Val
    def GetStonesInGame(self, Player):
        if Player == 1:
            return self.StonesInGame1
        else:
            return self.StonesInGame2
    def AddStoneToGame(self, Player):
        if Player == 1:
            self.StonesInGame1 += 1
            self.StonesInHand1 -= 1
        else:
            self.StonesInGame2 += 1
            self.StonesInHand2 -= 1
    def RemoveStone(self, Player):
        if Player == 1:
            self.StonesInGame1 -= 1
        else:
            self.StonesInGame2 -= 1
    def CopyState(self):
        b = np.copy(self.Board)
        obj = State_Class()
        obj.Board = b
        obj.StonesInGame1 = self.StonesInGame1
        obj.StonesInGame2 = self.StonesInGame2
        obj.StonesInHand1 = self.StonesInHand1
        obj.StonesInHand2 = self.StonesInHand2
        return obj

@njit
def GetMills():
    return [
    [0,1,2],
    [3,4,5],
    [10,11,12],
    [13,14,15],
    [0,6,13],
    [3,7,10],
    [5,8,12],
    [2,9,15]
    ]


#move notation:
#     0          1          2
#           3    4    5      
#     6     7         8     9
#          10   11    12    
#     13        14          15

@njit
def GetPlayerStones(State, P):
    p = []
    for x in range(len(State.Board)):
        if State.Board[x] == P:
            p.append(x)
    return p

#@njit
def WillFormMill(S, Player, Move):
    State = S.CopyState()
    if State.GetStonesInHand(Player) > 0:
        State.Board[Move[1]] = Player
    else:
        IsLegal = True
        if not ((State.Board[Move[0]] == Player) and (State.Board[Move[1]] == 0)):
            IsLegal = False
        if State.GetStonesInGame(Player) > 0: #since flying rule is not permitted in 6-men this will never be false, in 9-men change to >3
            if not IsAdjacent(Move[0], Move[1]):
                IsLegal = False
        if not IsLegal:
            raise Exception("illegal move.")
        else:
            State.Board[Move[0]] = 0
            State.Board[Move[1]] = Player #do the move
    #check if a new mill formed
    NewMill = MillAtPos(State, Move[1])
    return NewMill

@njit
def MillAtPos(State,p):
    Player = State.Board[p]
    for Mill in GetMills():
        if p in Mill:
            MilHere = True
            for pos in Mill:
                if not (State.Board[pos] == Player):
                    MilHere = False
                    break
            if MilHere:
                return True
    return False

@njit
def SemiMillAtPos(State,p, Player, Orig_p=None): #2 out of 3 are full by same player
    for Mill in GetMills():
        if p in Mill:
            NumPlayer = 0
            for pos in Mill:
                if (State.Board[pos] == Player):
                    if not (pos == Orig_p):
                        NumPlayer += 1
                elif (State.Board[pos] == -Player):
                    NumPlayer = 0
                    break
            if NumPlayer == 2:
                return True
    return False
@njit
def IsAdjacent(p1, p2): #will return false if p1 == p2
    return (p2 in GetConnections(p1))


@njit
def GetConnections(p):
    Connections = [
    [1,6],
    [0,2,4],
    [1,9],
    [4,7],
    [1,3,5],
    [4,8],
    [0,7,13],
    [3,6,10],
    [5,9,12],
    [2,8,15],
    [7,11],
    [10,12,14],
    [11,8],
    [6,14],
    [11,13,15],
    [9,14]
    ]
    return Connections[p]

            
# move: (source, destination, removelocation (during mill))
NewPiece = -1

@njit
def PlayMove(S, Player, Move): #Player = 1 (starting player), -1 (2nd player)
    State = S.CopyState()
    if State.GameOver == True:
        return State
    if State.GetStonesInHand(Player) > 0:
        State.Board[Move[1]] = Player
        State.AddStoneToGame(Player)
    else:
        IsLegal = True
        if not ((State.Board[Move[0]] == Player) and (State.Board[Move[1]] == 0)):
            IsLegal = False
        if State.GetStonesInGame(Player) > 0: #since flying rule is not permitted in 6-men this will never be false, in 9-men change to >3
            if not IsAdjacent(Move[0], Move[1]):
                IsLegal = False
        if not IsLegal:
            raise Exception("illegal move.")
        else:
            State.Board[Move[0]] = 0
            State.Board[Move[1]] = Player #do the move
    #check if a new mill formed
    NewMill = MillAtPos(State, Move[1])
    if NewMill: #remove one of opponent's piece
        State.Board[Move[2]] = 0
        State.RemoveStone(-Player)
        if State.GetStonesInGame(-Player) < 3:
            State.GameOver = True
    return State