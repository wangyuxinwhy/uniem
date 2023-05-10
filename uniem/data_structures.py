from dataclasses import dataclass
from enum import Enum


class RecordType(str, Enum):
    PAIR = 'pair'
    TRIPLET = 'triplet'
    SCORED_PAIR = 'scored_pair'


@dataclass(slots=True)
class PairRecord:
    text: str
    text_pos: str


@dataclass(slots=True)
class TripletRecord:
    text: str
    text_pos: str
    text_neg: str


@dataclass(slots=True)
class ScoredPairRecord:
    text: str
    text_pos: str
    score: float
