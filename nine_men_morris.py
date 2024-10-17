#6men game
import copy
import numba
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
@numba.jitclass(spec)
class State_Class():
    def __init__(self):
        self.Board = np.zeros(24, dtype=np.int8)
        self.StonesInHand1 = 9
        self.StonesInHand2 = 9
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
    def Arr(self):
        b = [self.Board[x] for x in range(24)]
        b.append(self.StonesInGame1)
        b.append(self.StonesInGame2)
        b.append(self.StonesInHand1)
        b.append(self.StonesInHand2)
        b.append(self.GameOver)
        return b

@njit
def SameState(s1, s2):
    if s1.StonesInGame1 != s2.StonesInGame1:
        return False
    if s1.StonesInGame2 != s2.StonesInGame2:
        return False
    if s1.StonesInHand1 != s2.StonesInHand1:
        return False
    if s1.StonesInHand2 != s2.StonesInHand2:
        return False
    if s1.GameOver != s2.GameOver:
        return False
    if not (s1.Board == s2.Board).all():
        return False
    return True
@njit
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

@njit
def GetMills():
    return [
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

#move notation:
#     00         01          02
#          03    04    05      
#             06 07 08
#     09   10 11    12 13    14
#             15 16 17
#          18    19    20    
#     21         22          23

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
def SemiMillAtPos(State,p, Player, Orig_p=-1): #2 out of 3 are full by same player
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
        rsn = ""
        if not ((State.Board[Move[0]] == Player) and (State.Board[Move[1]] == 0)):
            IsLegal = False
            rsn = "start pos not player or end pos not empty"
        if State.GetStonesInGame(Player) > 3: #dont check for adjacency when in flying mode, any stone can go anywhere
            if not IsAdjacent(Move[0], Move[1]):
                IsLegal = False
                rsn = "not adjacent"
        if not IsLegal:
            print("illegal move. "+rsn)
            raise Exception("illegal move. ")
        else:
            State.Board[Move[0]] = 0
            State.Board[Move[1]] = Player #do the move
    #check if a new mill formed
    NewMill = MillAtPos(State, Move[1])
    if NewMill: #remove one of opponent's piece
        State.Board[Move[2]] = 0
        State.RemoveStone(-Player)
        if State.GetStonesInGame(-Player)+State.GetStonesInHand(-Player) < 3:
            State.GameOver = True
    return State
