from enum import Enum

class Action(Enum):
    NULL = 0
    REMOVED = 1
    ADDED = 2
    SHIFTED = 3
    SHIFTED_SMALL = 4