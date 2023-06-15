from dataclasses import dataclass
from enum import Enum
from typing import Any


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
    sentence1: str
    sentence2: str
    label: float


record_type_cls_map: dict[RecordType, Any] = {
    RecordType.PAIR: PairRecord,
    RecordType.TRIPLET: TripletRecord,
    RecordType.SCORED_PAIR: ScoredPairRecord,
}
