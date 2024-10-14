
import socket

NUM_ENTITIES = 40

TYPE_DEAD = 0
TYPE_PLAYER_SHIP = 1
TYPE_FUEL_CAN = 2
TYPE_PLAYER_SHOT = 3
TYPE_ENEMY_SHOT = 4
TYPE_FLAGSHIP = 5
TYPE_SLICER = 6
TYPE_ENEMY_1 = 7
TYPE_ENEMY_2 = 8
TYPE_ENEMY_3 = 9
TYPE_ENEMY_4 = 10
TYPE_ENEMY_5 = 11
TYPE_ENEMY_6 = 12
TYPE_ENEMY_7 = 13
TYPE_ENEMY_8 = 14
TYPE_SOLAR_WASTER = 15
TYPE_COUNT = 16
TYPE_LETTERS = " PC^vF*abcdefghX"

def isEnemyType(entityType):
    return entityType is not None and entityType >= TYPE_ENEMY_1 and entityType <= TYPE_ENEMY_8

gEntityTypeToEntities = [set() for i in range(TYPE_COUNT)]

class Entity:
    def __init__(self):
        self.entityType = TYPE_DEAD
        self.x = 0
        self.y = 0
        self.dx = 0
        self.dy = 0
        self.history = []

    def update(self, entityType, x, y):
        if entityType == TYPE_DEAD and self.entityType != TYPE_DEAD:
            gEntityTypeToEntities[self.entityType].remove(self)
        self.entityType = entityType
        self.x = x
        self.y = y
        if self.entityType == TYPE_DEAD:
            self.history = []
            self.x = 0
            self.y = 0
            self.dx = 0
            self.dy = 0
        else:
            gEntityTypeToEntities[self.entityType].add(self)
            self.history.append( (self.x, self.y) )
            while len(self.history) > 10:
                self.history.pop(0)
            if len(self.history) == 10:
                self.dx = self.x - self.history[0][0]
                self.dy = self.y - self.history[0][1]

    def __repr__(self):
        if self.entityType is None:
            return "(?)"
        else:
            return "(%s,%d,%d,%d,%d)" % (TYPE_LETTERS[self.entityType], self.x, self.y, self.dx, self.dy)

gEntities = [Entity() for i in range(NUM_ENTITIES)]

def main():
    # To game.
    outputSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    outputSocket.connect( ("localhost", 5005) )

    # From game.
    inputSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    inputSocket.connect( ("localhost", 4004) )

    # create_connection( (host, port) )

    outputSocketFile = outputSocket.makefile("wb")
    inputSocketFile = inputSocket.makefile("rb")

    outputSocketFile.write(b"72\r")
    outputSocketFile.flush()

    initialSection = True

    while True:
        line = ""
        while True:
            # print("<")
            ch = inputSocketFile.read(1).decode("ascii")
            # print(">", repr(ch), repr(line), len(line))
            if ch == "\r" or ch == "\n":
                if len(line) > 0:
                    break
            else:
                line += ch
        index, param1, param2, param3 = map(int, line.split(","))
        #print(index, param1, param2, param3)
        if index < 128:
            gEntities[index].update(param1, param2, param3)
            if isEnemyType(param1) and param3 < 10:
                initialSection = False
        elif index == 128:
            score = param1
            shipsLeft = param2
            abmsLeft = param3

            if False:
                print("----------------------------")
                print("State:")
                for entityType in range(1, TYPE_COUNT):
                    line = "    %s: " % TYPE_LETTERS[entityType]
                    for entity in gEntityTypeToEntities[entityType]:
                        line += repr(entity) + " "
                    print(line)
            player = list(gEntityTypeToEntities[TYPE_PLAYER_SHIP])[0]
            candidateTargets = [entity for entity in gEntities
                                if isEnemyType(entity.entityType) or
                                (entity.entityType == TYPE_FUEL_CAN and entity.y <= player.y)]
            # Lower number is better.
            candidateTargets.sort(key=lambda entity:
                                  (entity.entityType != TYPE_FUEL_CAN, -entity.y))
            print(candidateTargets)
            targetEnemy = candidateTargets[0] if candidateTargets else None

            fire = False
            info = ""
            action = ""
            if targetEnemy is None:
                # Center ourselves.
                target = 64
            else:
                dx = player.x - targetEnemy.x
                dy = player.y - targetEnemy.y
                if dy < 8 and targetEnemy.entityType != TYPE_FUEL_CAN:
                    # print("dy", dy, "---------------------------------------------")
                    if targetEnemy.x < player.x:
                        target = targetEnemy.x + 15
                    else:
                        target = targetEnemy.x - 15
                else:
                    mult = 1 if targetEnemy.dy > 0 else 5
                    target = targetEnemy.x
                    if dy != 0:
                        target += mult * targetEnemy.dx * 5 // dy
                    if abs(target - player.x) <= 3:
                        fire = True
                info = "%s %s %s %s" % (player, targetEnemy, dx, dy)
            if target < player.x - 1:
                action += "<"
            elif target > player.x + 1:
                action += ">"
            if fire and not initialSection:
                action += "F"
            # print(info, abs(target - player.x), action)

            outputSocketFile.write((action + "\r").encode("ascii"))
            outputSocketFile.flush()
        elif index == 129:
            print("Game over")
            break
        else:
            print("Unknown index %d" % index)
            break

main()
