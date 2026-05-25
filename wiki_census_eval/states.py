from __future__ import annotations

from typing import Iterable, Optional, Set


POSTAL_TO_FIPS = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}


def parse_state_filters(values: Optional[Iterable[str]]) -> Set[str]:
    if values is None:
        return set()
    state_fips: Set[str] = set()
    for value in values:
        for part in value.split(","):
            token = part.strip().upper()
            if not token:
                continue
            if token in POSTAL_TO_FIPS:
                state_fips.add(POSTAL_TO_FIPS[token])
                continue
            if token.isdigit():
                state_fips.add(token.zfill(2))
                continue
            raise ValueError(
                f"Unknown state filter {part!r}; use postal abbreviations like AL "
                "or FIPS codes like 01."
            )
    return state_fips
