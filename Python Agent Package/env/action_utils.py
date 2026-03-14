"""
Shared direction constants. These **must** match the Java constants:

    0 = UP
    1 = RIGHT
    2 = DOWN
    3 = LEFT
    4 = NEUTRAL

Use these everywhere in your agent code.
"""

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NEUTRAL = -1

ALL_DIRECTIONS = [UP, RIGHT, DOWN, LEFT, NEUTRAL]
MOVE_DIRECTIONS = [UP, RIGHT, DOWN, LEFT]
