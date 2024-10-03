from enum import Enum
from collections import namedtuple
import re

DisplacementFileSections = Enum('DisplacementFileSections', [
    'GEO_DELTA',
    'VIB_DELTA',
    'OCC_DELTA',
    'CONSTRAIN'
])

LOOP_START_MARKER = 'LOOP_START'
LOOP_END_MARKER = 'LOOP_END'

LoopMarkerLine = namedtuple('LoopMarkerLine', ['type'])
SearchHeaderLine = namedtuple('SearchHeaderLine', ['label'])
SectionLine = namedtuple('SectionLine', ['section', 'line'])

SEARCH_HEADER_PATTERN = re.compile(r"^==\s+(?i:search)\s+(.*)$")
SECTION_HEADER_PATTERN = re.compile(r"^(GEO_DELTA|VIB_DELTA|OCC_DELTA|CONSTRAIN)$")


GEO_LINE_PATTERN = r"^(?P<label>\w+)\s+(?P<which>L\(\d+(-\d+)?\)|\d+(\s+\d+)*)\s+(?P<dir>[a-zA-Z]+(?:\[[^\]]+\]|\([^\)]+\))?)\s*=\s*(?P<start>-?\d+(\.\d+)?)(?:\s+(?P<stop>-?\d+(\.\d+)?))(?:\s+(?P<step>-?\d+(\.\d+)?))?$"

TEST_LINES_GEOMETRY = {
    "Ir L(1-6) z = -0.05 0.05 0.01": ('Ir', 'L(1-6)', 'z', -0.05, 0.05, 0.01),
    "Si = -0.005 0.005 0.0005": ('Si', None, None, -0.005, 0.005, 0.0005),
    "Ir L(1-6) xy[1 0] = -0.03 0.03 0.01": ('Ir', 'L(1-6)', 'xy[1 0]', -0.03, 0.03, 0.01),
    "Abc L(1-6) xy = -0.03 0.03 0.01": ('Abc', 'L(1-6)', 'xy', -0.03, 0.03, 0.01),
    "Ir L(1-6) z = -0.03 0.03 0.01": ('Ir', 'L(1-6)', 'z', -0.03, 0.03, 0.01),
    "Au 1 2 4 x = -0.03 0.03 0.01": ('Au', '1 2 4', 'x', -0.03, 0.03, 0.01),
    "B 1 2 4 abc = -0.03 0.03 0.01" : ('B', '1 2 4', 'abc', -0.03, 0.03, 0.01),
    "Cd L(1-6) azi(ab[c1 c2]) = -0.03 0.03 0.01" : ('Cd', 'L(1-6)', 'azi(ab[c1 c2])', -0.03, 0.03, 0.01),
    "E 5 ab[n1 n2] = -0.005 0": ('E', '5', 'ab[n1 n2]', -0.005, 0.005, None),
}