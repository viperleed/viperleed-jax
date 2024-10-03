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


GEO_LINE_PATTERN = re.compile(r"^(?P<label>\w+)\s+(?P<which>L\(\d+(-\d+)?\)|"
                              r"\d+(\s+\d+)*)\s+(?P<dir>[a-zA-Z]+(?:\[[^\]]+\]"
                              r"|\([^\)]+\))?)\s*=\s*(?P<start>-?\d+(\.\d+)?)"
                              r"(?:\s+(?P<stop>-?\d+(\.\d+)?))"
                              r"(?:\s+(?P<step>-?\d+(\.\d+)?))?$")
