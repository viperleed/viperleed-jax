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

GEO_LINE_PATTERN = re.compile(
    r"^(?P<label>\w+)"
    r"(?:\s+(?P<which>L\(\d+(-\d+)?\)|\d+(\s+\d+)*))?"
    r"\s+(?P<dir>[a-zA-Z]+(?:\[[^\]]+\]|\([^\)]+\))?)"
    r"\s*=\s*(?P<start>-?\d+(\.\d+)?)"
    r"(?:\s+(?P<stop>-?\d+(\.\d+)?))?"
    r"(?:\s+(?P<step>-?\d+(\.\d+)?))?$"
)
VIB_LINE_PATTERN = re.compile(
    r"^(?P<label>\w+)(?:\s+(?P<which>L\(\d+(-\d+)?\)|\d+(\s+\d+)*))?"
    r"\s*=\s*(?P<start>-?\d+(\.\d+)?)"
    r"(?:\s+(?P<stop>-?\d+(\.\d+)?)"
    r"(?:\s+(?P<step>-?\d+(\.\d+)?))?)?$"
)

def match_geo_line(line):
    match = GEO_LINE_PATTERN.match(line)
    if match is None:
        return None
    label = match.group('label')
    which = match.group('which') # optional, can be None
    dir = match.group('dir')
    start = float(match.group('start'))
    stop = float(match.group('stop')) if match.group('stop') is not None else None
    step = float(match.group('step')) if match.group('step') is not None else None
    return label, which, dir, start, stop, step

def match_vib_line(line):
    """Match and parse a VIB_DELTA line, returning the values as floats."""
    match = VIB_LINE_PATTERN.match(line)
    if match is None:
        return None

    label = match.group('label')
    which = match.group('which')
    start = float(match.group('start'))
    stop = float(match.group('stop')) if match.group('stop') is not None else None
    step = float(match.group('step')) if match.group('step') is not None else None

    return label, which, start, stop, step
