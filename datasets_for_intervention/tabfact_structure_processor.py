"""
tabfact_structure_processor.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TabFact-specific Tool and StructureProcessor.

Mediator for TabFact is a DSL query string such as:
    eq{count{filter_eq{filter_greater{all_rows; total; 30}; style; jive}}; 2}=True

TabFactTool
-----------
Deterministic function: executes the DSL query on the sample's table and returns
the boolean result (True / False). Analogous to RiceChemTool.calculate_score.

tool_mode="simple"  : ARGS contain {"query": "<dsl expression>"}
tool_mode="none"    : model outputs "Execution Result: True/False" directly

TabFactStructureProcessor
--------------------------
Handles:
  - extract_mediator    : verifier query string from completion
  - extract_final_answer: True/False from completion
  - extract_tool_args   : {"query": str} from tool call ARGS (simple mode)
  - compare_structures  : exact string comparison (canonical form)
  - check_generation_format_mistakes: is there preamble before "Verifier Query:"
  - extract_columns_values: parse AST to collect field names and values referenced
  - set_match           : set-equality check over columns ∪ values between two queries
"""

from __future__ import annotations

import io
import json
import re
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

from datasets_for_intervention.tabfact_dsl_engine import Call, Literal, TabFactEngine


# ============================================================
# TabFactTool
# ============================================================

class TabFactTool:
    """
    Deterministic table-query executor for TabFact.

    The 'score' (target) for TabFact is the boolean execution result of the
    DSL expression over the sample's table.

    Interface mirrors RiceChemTool so it plugs into the same pipeline:
      - name       : str
      - spec       : dict (JSON schema for the model)
      - spec_json(): str
      - validate_args(args): bool
      - calculate_score(args, sample_meta): bool | None
    """

    name: str = "check_query"

    def __init__(self, engine: TabFactEngine) -> None:
        self.engine = engine

    @property
    def spec(self) -> Dict[str, Any]:
        return {
            "title": self.name,
            "type": "object",
            "description": (
                "Execute a DSL verifier query over a table and return the boolean result. "
                "The query MUST end with =True or =False."
            ),
            "returns": {
                "type": "boolean",
                "description": "True if the logical statement holds, False otherwise.",
            },
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A fully-formed DSL expression ending with =True or =False, "
                        "e.g. eq{count{filter_eq{all_rows; style; jive}}; 2}=True"
                    ),
                }
            },
            "required": ["query"],
        }

    def spec_json(self) -> str:
        return json.dumps(self.spec, ensure_ascii=False)

    def validate_args(self, args: Any) -> bool:
        """
        Type-based validation of tool arguments (not mode-based).
        Accepts {"query": str} where query ends with =True or =False.
        """
        if not isinstance(args, dict):
            return False
        q = args.get("query")
        if not isinstance(q, str) or not q:
            return False
        return q.endswith("=True") or q.endswith("=False")

    def calculate_score(
        self, args: Dict[str, Any], sample_meta: Dict[str, Any]
    ) -> Optional[bool]:
        """
        Execute the DSL query in args["query"] on sample_meta['table_html_csv'].

        Returns True | False | None (None on parse/execution error).
        """
        if not self.validate_args(args):
            return None

        query = args["query"]
        table_content = sample_meta.get("table_html_csv")
        if not table_content:
            return None

        df = _parse_table_csv(table_content)
        if df is None or df.empty:
            return None

        result = self.engine.execute(query, df)
        return result.final if result.executable else None


# ============================================================
# TabFactStructureProcessor
# ============================================================

class TabFactStructureProcessor:
    """
    Handles parsing, validation, and comparison of TabFact DSL mediators.

    Mediator canonical form: a string like
        eq{count{filter_eq{filter_greater{all_rows; total; 30}; style; jive}}; 2}=True

    compare_structures uses EXACT STRING MATCH, which is correct for:
      - classify_generation  (predicted == gold?)
      - mediator_tool_match  (text-parsed == tool-parsed?)

    Jaccard Index (over extracted columns ∪ values) is provided separately via
    set_match() and is used as an additional evaluation metric.
    """

    # Matches: "Verifier Query: <expression>" at start of line (case-insensitive)
    QUERY_LINE_RE = re.compile(
        r"(?im)^verifier\s+query\s*:\s*(.+)$"
    )

    # Matches: "Execution Result: True/False" at start of line
    EXEC_LINE_RE = re.compile(
        r"(?im)^execution\s+result\s*:\s*(true|false)\s*$"
    )

    # Matches tool call ARGS block: ARGS: {...}
    TOOL_ARGS_BLOCK_RE = re.compile(
        r"(?is)\bARGS\s*:\s*(?P<block>.+?)(?:\n\s*\n|$)"
    )

    # Matches {"query": "<dsl>"} inside ARGS
    TOOL_QUERY_STRING_RE = re.compile(
        r'(?is)"query"\s*:\s*"(?P<query>[^"]+)"'
    )

    def __init__(self, engine: TabFactEngine) -> None:
        self.engine = engine

    # ------------------------------------------------------------------
    # Core parsing methods (required by Intervention / Evaluation)
    # ------------------------------------------------------------------

    def extract_mediator(self, completion_text: str) -> Optional[str]:
        """
        Extract the DSL verifier query from a full completion.

        Looks for the first line matching "Verifier Query: <expr>".
        Returns the query string if syntactically valid, else None.
        """
        m = self.QUERY_LINE_RE.search(completion_text or "")
        if not m:
            return None
        query = m.group(1).strip()
        if not self._is_valid_query(query):
            return None
        return query

    def extract_final_answer(
        self, text: str, short_completion: bool = False
    ) -> Optional[bool]:
        """
        Extract the boolean execution result from a completion.

        In full mode: looks for "Execution Result: True/False".
        In short_completion mode: the text is expected to be just "True" or "False"
        (the tail the model generates after the assistant prefix in interventions).
        """
        # Full completion: look for the "Execution Result:" line
        m = self.EXEC_LINE_RE.search(text or "")
        if m:
            return m.group(1).strip().lower() == "true"

        # Short completion: model output is only the verdict
        if short_completion:
            t = (text or "").strip().lower()
            if t in ("true", "true.", "true!"):
                return True
            if t in ("false", "false.", "false!"):
                return False

        return None

    def extract_tool_args(
        self, text: str, short_completion: bool = False
    ) -> Optional[Dict[str, str]]:
        """
        Extract tool arguments from a completion for tool_mode='simple'.

        Expected format in completion:
            Final tool call:
               TOOL: execute_query
               ARGS: {"query": "<dsl expression>"}

        In short_completion mode the text is the tail after "Final tool call:\n".

        Returns {"query": "<dsl>"} or None on parse failure.
        """
        raw = text or ""

        if short_completion:
            block = _clean_tool_text(raw)
        else:
            m = self.TOOL_ARGS_BLOCK_RE.search(raw)
            if not m:
                return None
            block = _clean_tool_text(m.group("block"))

        if not block:
            return None

        ms = self.TOOL_QUERY_STRING_RE.search(block)
        if not ms:
            return None

        query = ms.group("query").strip()
        if not self._is_valid_query(query):
            return None

        return {"query": query}

    def compare_structures(self, a: Any, b: Any) -> Optional[int]:
        """
        Exact-string comparison of two DSL query strings.

        Returns:
            1     if a == b (strings identical)
            0     if a != b
            None  if either argument is None
        """
        if a is None or b is None:
            return None
        return 1 if a == b else 0

    def check_generation_format_mistakes(self, completion: str) -> bool:
        """
        Check for garbage in the XM part of the completion.

        Garbage = any text before "Verifier Query:" line.
        (We do NOT penalise anything after "Execution Result:" — tail garbage is
        handled by infer_completion returning None.)

        Returns True → garbage detected → generation_status = "error".
        """
        s = (completion or "").strip()
        if not s:
            return True

        # The completion must start immediately with "Verifier Query:"
        if not re.match(r"(?i)verifier\s+query\s*:", s):
            return True

        return False

    # ------------------------------------------------------------------
    # Jaccard / column-value extraction (for evaluation metrics)
    # ------------------------------------------------------------------

    def extract_columns_values(
        self, query: str
    ) -> Tuple[Set[str], Set[str]]:
        """
        Parse the DSL query and extract sets of column names and literal values.

        Column names are field-name arguments to:
            filter_*, hop, argmax, argmin, all_*, uniq, most_freq
        Values are the literal arguments that appear where a value is expected:
            filter_eq{C; field; VALUE}, hop{C; FIELD} -> FIELD is column
            all_eq{C; field; VALUE}, etc.

        Returns (columns: set[str], values: set[str]).
        Both sets use lower-cased, stripped strings for comparison.
        """
        columns: Set[str] = set()
        values: Set[str] = set()

        try:
            prog = self.engine.parser.parse_program(query)
            _collect_cols_vals(prog.expr, columns, values)
        except Exception:
            pass

        return columns, values

    def set_match(self, a: Optional[str], b: Optional[str]) -> Optional[int]:
        """
        Set-equality check over the union of (columns ∪ values) from two DSL queries.

        Returns 1 if the combined sets of column references and filter values are
        identical (order-independent), 0 if they differ, or None if either query
        is None or unparseable.

        Example: queries that differ only in operator (greater vs less) but reference
        the same columns and values will return 1.
        """
        if a is None or b is None:
            return None

        cols_a, vals_a = self.extract_columns_values(a)
        cols_b, vals_b = self.extract_columns_values(b)

        set_a = cols_a | vals_a
        set_b = cols_b | vals_b

        if not set_a and not set_b:
            # Both queries have no identifiers; fall back to exact string match
            return 1 if a == b else 0

        return 1 if set_a == set_b else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_valid_query(self, query: str) -> bool:
        """Return True iff query is a syntactically valid DSL program."""
        if not query:
            return False
        if not (query.endswith("=True") or query.endswith("=False")):
            return False
        try:
            self.engine.parser.parse_program(query)
            return True
        except Exception:
            return False


# ============================================================
# Module-level helpers
# ============================================================

def _parse_table_csv(table_content: str) -> Optional[pd.DataFrame]:
    """
    Parse the '#'-delimited CSV table string into a DataFrame.
    Returns None on parse failure.
    """
    try:
        df = pd.read_csv(
            io.StringIO(table_content),
            sep="#",
            header=0,
            dtype=str,
            on_bad_lines="skip",
        )
        return df
    except Exception:
        return None


def _clean_tool_text(s: str) -> str:
    """Remove markdown fences and unescape common escapes from tool ARGS block."""
    if not s:
        return ""
    s = s.strip()
    # Remove markdown code fences
    s = re.sub(r"(?is)^\s*```[a-z0-9_-]*\s*", "", s)
    s = re.sub(r"(?is)\s*```\s*$", "", s)
    return s.strip()


def _collect_cols_vals(
    node: Any, columns: Set[str], values: Set[str]
) -> None:
    """
    Recursively traverse an AST node and collect column names and values.

    Convention:
      - FIELD arguments (column names) are the 2nd arg (index 1) of:
            filter_eq, filter_not_eq, filter_greater, filter_greater_eq,
            filter_less, filter_less_eq, hop, argmax, argmin,
            all_eq, all_not_eq, all_greater, all_greater_eq, all_less, all_less_eq,
            uniq, most_freq, within, not_within, any_eq
      - VALUE arguments are the 3rd arg (index 2) of the filter_*/all_* family
      - For `hop{C; FIELD}`: FIELD is a column name (index 1)
    """
    if isinstance(node, Literal):
        return  # literals at top level are already handled by their parent Call

    if not isinstance(node, Call):
        return

    name = node.name  # already canonicalized by ProgramParser

    # Functions where arg[1] is a column name
    _FIELD_ARG1 = frozenset({
        "filter_eq", "filter_not_eq", "filter_greater",
        "filter_greater_eq", "filter_less", "filter_less_eq",
        "hop", "argmax", "argmin",
        "all_eq", "all_not_eq", "all_greater", "all_greater_eq",
        "all_less", "all_less_eq",
        "uniq", "most_freq",
        "within", "not_within", "any_eq",
    })

    # Functions where arg[2] is a value literal
    _VALUE_ARG2 = frozenset({
        "filter_eq", "filter_not_eq", "filter_greater",
        "filter_greater_eq", "filter_less", "filter_less_eq",
        "all_eq", "all_not_eq", "all_greater", "all_greater_eq",
        "all_less", "all_less_eq",
        "within", "not_within", "any_eq",
    })

    if name in _FIELD_ARG1 and len(node.args) >= 2:
        field_node = node.args[1]
        if isinstance(field_node, Literal):
            col = field_node.raw.strip().lower()
            if col and col not in ("all_rows", "true", "false"):
                columns.add(col)

    if name in _VALUE_ARG2 and len(node.args) >= 3:
        val_node = node.args[2]
        if isinstance(val_node, Literal):
            val = val_node.raw.strip().lower()
            # Skip numeric-looking tokens and special keywords
            if val and val not in ("true", "false", "all_rows"):
                try:
                    float(val.replace(",", ""))
                except ValueError:
                    values.add(val)

    # Recurse into all child nodes
    for child in node.args:
        _collect_cols_vals(child, columns, values)