
import socket, subprocess, time

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
TYPE_ENEMIES = set([
    TYPE_ENEMY_1,
    TYPE_ENEMY_2,
    TYPE_ENEMY_3,
    TYPE_ENEMY_4,
    TYPE_ENEMY_5,
    TYPE_ENEMY_6,
    TYPE_ENEMY_7,
    TYPE_ENEMY_8,
])

# Possible actions we can take in the game.
ACTIONS = [
        "",
        "<",
        ">",
        "F",
        "<F",
        ">F",
        # TODO add ABM (any digit).
]

def isEnemyType(entityType):
    return entityType is not None and entityType >= TYPE_ENEMY_1 and entityType <= TYPE_ENEMY_8

class EntityState:
    def __init__(self, entity_type, x, y):
        self.entity_type = entity_type
        self.x = x
        self.y = y

class GameState:
    def __init__(self):
        self.entities = [None for i in range(NUM_ENTITIES)]
        self.score = None
        self.ships_left = None
        self.abms_left = None
        self.game_over = False

    def get_entities_by_types(self, entity_types):
        return [entity for entity in self.entities if entity and entity.entity_type in entity_types]

    def set_entity_state(self, entity_index, entity_type, x, y):
        if entity_type == TYPE_DEAD:
            self.entities[entity_index] = None
        else:
            self.entities[entity_index] = EntityState(entity_type, x, y)

    def set_game_info(self, score, ships_left, abms_left):
        self.score = score
        self.ships_left = ships_left
        self.abms_left = abms_left

    def __repr__(self):
        return "GameState[%d entities, score %s, ships %s, abms %s]" % (
                sum(1 for entity in self.entities if entity),
                self.score, self.ships_left, self.abms_left)

class LiveGame:
    def __init__(self):
        self.state = GameState()

        args = [
            "/Applications/trs80gp.app/Contents/MacOS/trs80gp",
            # Model III.
            "-m3",
            # Run as fast as possible.
            "-turbo",
            # Disable floppy disk controller.
            "-dx",
            # Printer (output) port.
            "-p",
            "@4004",
            # Input port.
            "-ip",
            "@5005",
            # Cassette to launch.
            "remdc.cas",
        ]
        self.proc = subprocess.Popen(args)

        # Wait for it to start before we try to connect.
        time.sleep(0.5)

        # To game.
        outputSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        outputSocket.connect( ("localhost", 5005) )

        # From game.
        inputSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        inputSocket.connect( ("localhost", 4004) )

        # Make file objects.
        self.outputSocketFile = outputSocket.makefile("wb")
        self.inputSocketFile = inputSocket.makefile("rb")

    def start_new_game(self, seed):
        # Reset state.
        self.state = GameState()

        # Send seed.
        self.write_line(str(seed))

    def read_state(self):
        while True:
            index, param1, param2, param3 = self.read_line_values()
            if index < 128:
                self.state.set_entity_state(index, param1, param2, param3)
            elif index == 128:
                # End of state.
                self.state.set_game_info(param1, param2, param3)
                break
            elif index == 129:
                # Game over.
                self.state.game_over = True
                break

        return self.state

    # Blocks to get the next line from the game. Returns a tuple of integers.
    def read_line_values(self):
        line = ""

        # Must read the line ourselves because readline() does the wrong thing.
        while True:
            ch = self.inputSocketFile.read(1).decode("ascii")
            if ch == "\r" or ch == "\n":
                if len(line) > 0:
                    break
            else:
                line += ch

        return tuple(map(int, line.split(",")))

    def perform_action(self, action):
        self.write_line(ACTIONS[action])

    # Write the line to the game. The line must be a string and not include
    # any newline.
    def write_line(self, line):
        line += "\r"
        self.outputSocketFile.write(line.encode("ascii"))
        self.outputSocketFile.flush()

    def kill(self):
        self.proc.terminate()

