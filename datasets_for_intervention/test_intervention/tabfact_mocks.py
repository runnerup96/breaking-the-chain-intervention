import random


class TabFactDatasetMock:
    """
    Mock class for TabFactDataset with support for diverse intervention tests.
    """

    def __init__(self):
        self.mock_table_content = (
            "Rank#Athlete#Nation#Gold#Silver#Bronze#Event#Time\n"
            "1#Usain Bolt#Jamaica#8#0#1#100m#9.63\n"
            "2#Shawn Crawford#United States#1#2#0#200m#19.79\n"
            "3#Carl Lewis#United States#9#1#0#Long Jump#8.87"
        )

        # -------------------
        # Основные сэмплы
        # -------------------
        self.data = [
            # --- Base sample with filter_eq and hop ---
            {
                "idx": "mock_0@table1",
                "table_id": "table1.html.csv",
                "table_html_csv": self.mock_table_content,
                "statement": "Usain Bolt won more gold medals than Shawn Crawford.",
                "table_caption": "Olympic Medalists",
                "verifier_query_gt": (
                    "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
                    "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True"
                ),
                "label_gt": True,
                "distractors": {
                    "columns": [
                        "Rank", "Athlete", "Nation",
                        "Gold", "Silver", "Bronze",
                        "Event", "Time"
                    ],
                    "values": {
                        "athlete": ["Usain Bolt", "Shawn Crawford", "Carl Lewis"],
                        "nation": ["Jamaica", "United States", "Canada"],
                        "gold": ["8", "1", "9", "5", "3"],
                        "silver": ["0", "2", "1", "0", "4"],
                        "bronze": ["1", "0", "0", "2"],
                        "event": ["100m", "200m", "Long Jump", "400m"],
                        "time": ["9.63", "19.79", "8.87", "10.12"]
                    },
                    "entity_swaps": [
                        "Usain Bolt", "Shawn Crawford", "Carl Lewis",
                        "Jamaica", "United States", "Canada",
                        "8", "1", "9", "0", "2",
                        "100m", "200m", "Long Jump"
                    ]
                }
            },
            # --- Sample with argmax ---
            {
                "idx": "mock_1@table1",
                "table_id": "table1.html.csv",
                "table_html_csv": self.mock_table_content,
                "statement": "Carl Lewis has the most gold medals.",
                "table_caption": "Olympic Medalists",
                "verifier_query_gt": "eq{hop{argmax{all_rows; gold}; athlete}; Carl Lewis}=True",
                "label_gt": True,
                "distractors": {
                    "columns": [
                        "Rank", "Athlete", "Nation",
                        "Gold", "Silver", "Bronze",
                        "Event", "Time"
                    ],
                    "values": {
                        "athlete": ["Usain Bolt", "Shawn Crawford", "Carl Lewis"],
                        "nation": ["Jamaica", "United States", "Canada"],
                        "gold": ["8", "1", "9", "5", "3"],
                        "silver": ["0", "2", "1", "0", "4"],
                        "bronze": ["1", "0", "0", "2"],
                        "event": ["100m", "200m", "Long Jump", "400m"],
                        "time": ["9.63", "19.79", "8.87", "10.12"]
                    },
                    "entity_swaps": [
                        "Usain Bolt", "Shawn Crawford", "Carl Lewis",
                        "Jamaica", "United States", "Canada",
                        "8", "1", "9", "0", "2",
                        "100m", "200m", "Long Jump"
                    ]
                }
            },
            # --- Sample with aggregation (avg) ---
            {
                "idx": "mock_2@table1",
                "table_id": "table1.html.csv",
                "table_html_csv": self.mock_table_content,
                "statement": "Average time for US athletes is better than 15 seconds.",
                "table_caption": "Olympic Medalists",
                "verifier_query_gt": (
                    "less{avg{filter_eq{all_rows; nation; United States}; time}; 15}=True"
                ),
                "label_gt": True,
                "distractors": {
                    "columns": [
                        "Rank", "Athlete", "Nation",
                        "Gold", "Silver", "Bronze",
                        "Event", "Time"
                    ],
                    "values": {
                        "athlete": ["Usain Bolt", "Shawn Crawford", "Carl Lewis"],
                        "nation": ["Jamaica", "United States", "Canada"],
                        "gold": ["8", "1", "9", "5", "3"],
                        "silver": ["0", "2", "1", "0", "4"],
                        "bronze": ["1", "0", "0", "2"],
                        "event": ["100m", "200m", "Long Jump", "400m"],
                        "time": ["9.63", "19.79", "8.87", "10.12", "15.5", "20.0"]
                    },
                    "entity_swaps": [
                        "Usain Bolt", "Shawn Crawford", "Carl Lewis",
                        "Jamaica", "United States", "Canada",
                        "8", "1", "9", "0", "2",
                        "100m", "200m", "Long Jump",
                        "15", "10", "20"
                    ]
                }
            }
        ]

        # -------------------
        # Alternative questions/programs
        # -------------------
        self.table_id2alt_questions = {
            "table1.html.csv": [
                "Shawn Crawford is from Jamaica.",
                "Usain Bolt has exactly 1 gold medal.",
                "Carl Lewis competed in the 100m event.",
                "The fastest time belongs to a US athlete."
            ]
        }
        self.table_id2alt_programs = {
            "table1.html.csv": [
                "eq{Jamaica; hop{filter_eq{all_rows; athlete; Shawn Crawford}; nation}}=False",
                "eq{1; hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}}=False",
                "eq{100m; hop{filter_eq{all_rows; athlete; Carl Lewis}; event}}=False",
                "eq{United States; hop{argmin{all_rows; time}; nation}}=False"
            ]
        }

        # -------------------
        # Pre-generated Local Edits
        # -------------------
        self.sample_id2local_edits = {
            "mock_0@table1": [
                # Athlete swap
                "greater{hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}; hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}}=True",
                # Operator change
                "less{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True",
                "eq{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True",
                # Self-comparison
                "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}}=True",
                # Filter change
                "greater{hop{filter_eq{all_rows; nation; Jamaica}; gold}; hop{filter_eq{all_rows; nation; United States}; gold}}=True"
            ],
            "mock_1@table1": [
                "eq{hop{argmax{all_rows; gold}; athlete}; Usain Bolt}=True",
                "not_eq{hop{argmax{all_rows; gold}; athlete}; Carl Lewis}=True",
                "eq{hop{argmax{all_rows; silver}; athlete}; Carl Lewis}=True",
                "eq{hop{argmax{all_rows; gold}; nation}; Carl Lewis}=True",
                "eq{hop{argmax{all_rows; gold}; athlete}; Shawn Crawford}=True"
            ],
            "mock_2@table1": [
                "less{avg{filter_eq{all_rows; nation; Jamaica}; time}; 15}=True",
                "greater{avg{filter_eq{all_rows; nation; United States}; time}; 15}=True",
                "eq{avg{filter_eq{all_rows; nation; United States}; time}; 15}=True",
                "less{avg{filter_eq{all_rows; event; 100m}; time}; 15}=True",
                "less{sum{filter_eq{all_rows; nation; United States}; time}; 15}=True"
            ]
        }

    # -------------------
    # API
    # -------------------
    def get_random_alternate_question(self, sample: dict) -> str:
        """Returns a random alternative question for the sample's table."""
        table_id = sample['table_id']
        pool = self.table_id2alt_questions.get(table_id, [])
        if pool:
            return random.choice(pool)
        return sample['statement']

    def get_random_alternate_program(self, sample: dict) -> str:
        """Returns a random alternative program for the sample's table."""
        table_id = sample['table_id']
        pool = self.table_id2alt_programs.get(table_id, [])
        if pool:
            return random.choice(pool)

        # fallback
        orig_prog = sample['verifier_query_gt']
        if orig_prog.endswith("=True"):
            return orig_prog[:-len("=True")] + "=False"
        elif orig_prog.endswith("=False"):
            return orig_prog[:-len("=False")] + "=True"
        return orig_prog

    def get_random_local_edits(self, sample: dict, n: int = 3) -> list[str]:
        """Returns a random sample of n local edits for the given sample."""
        sample_id = sample['idx']
        pool = self.sample_id2local_edits.get(sample_id, [])
        if len(pool) < n:
            return pool + random.choices(pool, k=n - len(pool)) if pool else ["eq{1;0}=True"] * n
        return random.sample(pool, n)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        return self.data[i]
