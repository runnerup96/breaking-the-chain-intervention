"""
tabfact_mocks.py
~~~~~~~~~~~~~~~~
Mock TabFactDataset for unit tests.

New architecture keys per sample:
  idx, gold_query, mediator_query, gold_target,
  table_id, table_html_csv, statement, table_caption

sample_id2local_edits: {idx: [{"query": str, "expected_target": bool}]}
  - Only edits verified to execute to result != gold_target are stored.
  - gold_target is always True.

get_local_edits(sample, n=None) mirrors TabFactDataset.get_local_edits.
"""

import random
from copy import deepcopy


# -----------------------------------------------------------------------
# Table fixtures
# -----------------------------------------------------------------------
#
# Suffix semantics (TabFactEngine):
#   expr=True  →  final = (eval(expr) == True)
#   expr=False →  final = (eval(expr) == False)
#
# Sample 0: greater{Usain.gold; Shawn.gold}=True → 8>1=True → final=True  (gold)
# Sample 1: eq{argmax(gold).athlete; Carl Lewis}=True → True → final=True  (gold)
# Sample 2: less{avg{US.time}; 15}=True → 14.33<15=True → final=True  (gold)

_TABLE = (
    "rank#athlete#nation#gold#silver#bronze\n"
    "1#Usain Bolt#Jamaica#8#0#1\n"
    "2#Shawn Crawford#United States#1#2#0\n"
    "3#Carl Lewis#United States#9#1#0"
)

_TABLE_WITH_TIME = (
    "rank#athlete#nation#gold#silver#bronze#time\n"
    "1#Usain Bolt#Jamaica#8#0#1#9.63\n"
    "2#Shawn Crawford#United States#1#2#0#19.79\n"
    "3#Carl Lewis#United States#9#1#0#8.87"
)


def _s(idx, stmt, gq, table=_TABLE):
    return {
        "idx": idx,
        "table_id": "table1.html.csv",
        "table_html_csv": table,
        "statement": stmt,
        "table_caption": "Olympic Medalists",
        "gold_query": gq,
        "mediator_query": deepcopy(gq),
        "gold_target": True,
    }


class TabFactDatasetMock:
    """
    Mock TabFactDataset with new architecture keys.

    Three samples covering common DSL patterns:
      0 - comparison (greater / hop / filter_eq)
      1 - argmax + hop + eq
      2 - avg + filter_eq + less  (uses time column)

    Each pool has 4-5 verified local edits (all expected_target=False).
    """

    def __init__(self):
        self.data = [
            _s(
                "mock_0@table1",
                "Usain Bolt won more gold medals than Shawn Crawford.",
                (
                    "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
                    "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True"
                ),
            ),
            _s(
                "mock_1@table1",
                "Carl Lewis has the most gold medals.",
                "eq{hop{argmax{all_rows; gold}; athlete}; Carl Lewis}=True",
            ),
            _s(
                "mock_2@table1",
                "Average time for US athletes is under 15 seconds.",
                "less{avg{filter_eq{all_rows; nation; United States}; time}; 15}=True",
                table=_TABLE_WITH_TIME,
            ),
        ]

        # Each entry: {"query": str, "expected_target": bool}
        # All expected_target=False (verified to differ from gold_target=True).
        self.sample_id2local_edits = {
            # Sample 0: greater{Usain.gold; Shawn.gold}=True  (8>1=True, gold)
            "mock_0@table1": [
                # operator flip  ->  8<1=False, =True -> final=False
                {
                    "query": (
                        "less{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
                        "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True"
                    ),
                    "expected_target": False,
                },
                # entity swap  ->  1>8=False, =True -> final=False
                {
                    "query": (
                        "greater{hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}; "
                        "hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}}=True"
                    ),
                    "expected_target": False,
                },
                # suffix flip  ->  8>1=True, =False -> final=False
                {
                    "query": (
                        "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
                        "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=False"
                    ),
                    "expected_target": False,
                },
                # operator change to eq  ->  8==1=False, =True -> final=False
                {
                    "query": (
                        "eq{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
                        "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True"
                    ),
                    "expected_target": False,
                },
                # not_eq suffix flip  ->  8!=1=True, =False -> final=False
                {
                    "query": (
                        "not_eq{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
                        "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=False"
                    ),
                    "expected_target": False,
                },
            ],
            # Sample 1: eq{argmax(gold).athlete; Carl Lewis}=True (gold)
            "mock_1@table1": [
                # wrong target  ->  Carl Lewis != Usain Bolt -> False, =True -> False
                {
                    "query": "eq{hop{argmax{all_rows; gold}; athlete}; Usain Bolt}=True",
                    "expected_target": False,
                },
                # suffix flip  ->  True, =False -> False
                {
                    "query": "eq{hop{argmax{all_rows; gold}; athlete}; Carl Lewis}=False",
                    "expected_target": False,
                },
                # negation: Carl Lewis != Carl Lewis = False, =True -> False
                {
                    "query": "not_eq{hop{argmax{all_rows; gold}; athlete}; Carl Lewis}=True",
                    "expected_target": False,
                },
                # argmax silver: Shawn Crawford(2) != Carl Lewis -> False, =True -> False
                {
                    "query": "eq{hop{argmax{all_rows; silver}; athlete}; Carl Lewis}=True",
                    "expected_target": False,
                },
            ],
            # Sample 2: less{avg{US.time}; 15}=True  avg(19.79,8.87)=14.33 (gold)
            "mock_2@table1": [
                # operator flip  ->  14.33>15=False, =True -> False
                {
                    "query": "greater{avg{filter_eq{all_rows; nation; United States}; time}; 15}=True",
                    "expected_target": False,
                },
                # suffix flip  ->  True, =False -> False
                {
                    "query": "less{avg{filter_eq{all_rows; nation; United States}; time}; 15}=False",
                    "expected_target": False,
                },
                # equality  ->  14.33==15=False, =True -> False
                {
                    "query": "eq{avg{filter_eq{all_rows; nation; United States}; time}; 15}=True",
                    "expected_target": False,
                },
                # Jamaica avg=9.63, 9.63<9=False, =True -> False
                {
                    "query": "less{avg{filter_eq{all_rows; nation; Jamaica}; time}; 9}=True",
                    "expected_target": False,
                },
            ],
        }

    # ------------------------------------------------------------------
    # Public API  --  mirrors TabFactDataset
    # ------------------------------------------------------------------

    def get_local_edits(self, sample: dict, n: int = None) -> list:
        """
        Return verified local-edit entries for this sample.
        Each entry: {"query": str, "expected_target": bool}
        """
        pool = self.sample_id2local_edits.get(sample["idx"], [])
        if n is None or n >= len(pool):
            return list(pool)
        indices = random.sample(range(len(pool)), n)
        return [pool[i] for i in indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        return self.data[i]