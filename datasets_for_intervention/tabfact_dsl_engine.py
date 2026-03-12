from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import pandas as pd


class ParseError(Exception):
    """Raised when DSL cannot be parsed into an AST."""
    pass


class EvalError(Exception):
    """Raised when AST cannot be executed on a given table (type/arity/unknown op/etc)."""
    pass


@dataclass(frozen=True)
class Literal:
    raw: str


@dataclass(frozen=True)
class Call:
    name: str
    args: List[Any]  # List[Literal|Call]


@dataclass(frozen=True)
class Program:
    expr: Any        # Literal|Call
    expected: bool   # suffix bool from "=True/False"


Scalar = Union[str, float, bool]
Row = int


@dataclass(frozen=True)
class RowSet:
    rows: List[int]
    hint_field: Optional[str] = None

    def is_empty(self) -> bool:
        return len(self.rows) == 0

    def size(self) -> int:
        return len(self.rows)

    def top(self) -> int:
        if not self.rows:
            raise EvalError("top() on empty RowSet")
        return self.rows[0]

    def bottom(self) -> int:
        if not self.rows:
            raise EvalError("bottom() on empty RowSet")
        return self.rows[-1]


@dataclass(frozen=True)
class ScalarList:
    values: List[Scalar]

    def is_empty(self) -> bool:
        return len(self.values) == 0


@dataclass(frozen=True)
class ExecResult:
    """
    - final: the result of the whole program (taking the "=True/False" suffix into account), or None if execution failed
    - expr_value: the boolean value of the expression (before comparison with the suffix), or None
    - expected: the suffix, or None
    - error: a string with the reason if execution failed
    - executable: flag (final != None)
    """
    final: Optional[bool]
    expr_value: Optional[bool]
    expected: Optional[bool]
    error: Optional[str]
    executable: bool


@dataclass(frozen=True)
class FunctionSpec:
    canonical: str
    aliases: Tuple[str, ...]
    min_args: int
    max_args: Optional[int]  # None => no upper bound
    variadic: bool = False


class FunctionRegistry:
    """
    Language registry:
    - maps aliases to canonical names
    - knows the expected arity (min/max) for arity repair and validation
    """

    def __init__(self) -> None:
        self._specs: Dict[str, FunctionSpec] = {}
        self._alias_to_canon: Dict[str, str] = {}
        self._init_specs()

    def _init_specs(self) -> None:
        # Canonical names and arities are taken from what actually appears in bootstrap_full.json.
        specs = [
            # comparisons
            FunctionSpec("eq", ("eq", "equal"), 2, 2),
            FunctionSpec("not_eq", ("not_eq", "ne", "neq"), 2, 2),
            FunctionSpec("greater", ("greater", "gt", "more_than"), 2, 2),
            FunctionSpec("less", ("less", "lt"), 2, 2),

            # boolean logic
            FunctionSpec("and", ("and",), 2, None, variadic=True),
            FunctionSpec("or", ("or",), 2, None, variadic=True),
            FunctionSpec("not", ("not",), 1, 1),

            # rowset filters
            FunctionSpec("filter_eq", ("filter_eq", "filter_equal"), 3, 3),
            FunctionSpec("filter_not_eq", ("filter_not_eq", "filter_ne"), 3, 3),
            FunctionSpec("filter_greater", ("filter_greater",), 3, 3),
            FunctionSpec("filter_greater_eq", ("filter_greater_eq",), 3, 3),
            FunctionSpec("filter_less", ("filter_less",), 3, 3),
            FunctionSpec("filter_less_eq", ("filter_less_eq",), 3, 3),

            # row navigation / selection
            FunctionSpec("hop", ("hop",), 2, 2),
            FunctionSpec("argmax", ("argmax",), 1, 2),  # a rare 1-arg form appears
            FunctionSpec("argmin", ("argmin",), 1, 2),
            FunctionSpec("top", ("top",), 1, 1),
            FunctionSpec("bottom", ("bottom",), 1, 1),

            # aggregates (usually 2-arg: RowSet; Field)
            FunctionSpec("count", ("count",), 1, 1),
            FunctionSpec("sum", ("sum",), 1, 2),
            FunctionSpec("avg", ("avg",), 1, 2),
            FunctionSpec("max", ("max",), 2, 2),
            FunctionSpec("min", ("min",), 2, 2),
            FunctionSpec("uniq", ("uniq",), 2, 2),
            FunctionSpec("most_freq", ("most_freq",), 2, 2),
            FunctionSpec("half", ("half",), 1, 1),

            # quantifiers / membership
            FunctionSpec("within", ("within",), 3, 3),
            FunctionSpec("not_within", ("not_within",), 3, 3),
            FunctionSpec("any_eq", ("any_eq",), 3, 3),
            FunctionSpec("any", ("any",), 1, 1),
            FunctionSpec("none", ("none",), 1, 1),
            FunctionSpec("only", ("only",), 1, 1),
            FunctionSpec("zero", ("zero",), 1, 1),

            # all_* (usually 3-arg: RowSet; Field; Value, but rare 2-arg forms also exist)
            FunctionSpec("all_eq", ("all_eq",), 2, 3),
            FunctionSpec("all_not_eq", ("all_not_eq",), 3, 3),
            FunctionSpec("all_greater", ("all_greater",), 2, 3),
            FunctionSpec("all_greater_eq", ("all_greater_eq",), 3, 3),
            FunctionSpec("all_less", ("all_less",), 3, 3),
            FunctionSpec("all_less_eq", ("all_less_eq",), 3, 3),

            # arithmetic
            FunctionSpec("diff", ("diff",), 2, 2),
            FunctionSpec("add", ("add",), 2, 2),

            # order / positional predicates (table order)
            FunctionSpec("before", ("before",), 2, 2),
            FunctionSpec("after", ("after",), 2, 2),
            FunctionSpec("first", ("first",), 2, 2),
            FunctionSpec("second", ("second",), 2, 2),
            FunctionSpec("third", ("third",), 2, 2),
            FunctionSpec("fourth", ("fourth",), 2, 2),
            FunctionSpec("fifth", ("fifth",), 2, 2),
            FunctionSpec("last", ("last",), 2, 2),

            # rare
            FunctionSpec("rank", ("rank",), 2, 2),
        ]

        for spec in specs:
            self._specs[spec.canonical] = spec
            for a in spec.aliases:
                self._alias_to_canon[a] = spec.canonical

    def canonicalize(self, name: str) -> str:
        n = name.strip()
        if n in self._alias_to_canon:
            return self._alias_to_canon[n]
        # if a new function suddenly appears, let it fail as unknown
        return n

    def get_spec(self, canonical: str) -> Optional[FunctionSpec]:
        return self._specs.get(canonical)

    def arity_bounds(self, canonical: str) -> Tuple[int, Optional[int], bool]:
        spec = self.get_spec(canonical)
        if spec is None:
            raise EvalError(f"Unknown function: {canonical}")
        return spec.min_args, spec.max_args, spec.variadic


class NumberParser:
    """
    Converts a string to a number when it is reasonable to do so.
    Supports:
    - M:SS(.xx) -> seconds
    - mixed fraction: a - b / c
    - fraction: b / c
    - fallback: the first number in the string ("89.7 fm" -> 89.7)
    """

    _re_time = re.compile(r"^\s*(\d+)\s*:\s*(\d+(?:\.\d+)?)\s*$")
    _re_mixed = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*/\s*(\d+)\s*$")
    _re_frac = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
    _re_first_num = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    def parse_literal(self, s: str) -> Optional[float]:
        if s is None:
            return None
        t = str(s).strip()
        if not t:
            return None

        # remove thousands separators
        t2 = t.replace(",", "")

        m = self._re_time.match(t2)
        if m:
            mm = float(m.group(1))
            ss = float(m.group(2))
            return mm * 60.0 + ss

        m = self._re_mixed.match(t2)
        if m:
            a = float(m.group(1))
            b = float(m.group(2))
            c = float(m.group(3))
            if c == 0:
                return None
            return a + (b / c)

        m = self._re_frac.match(t2)
        if m:
            b = float(m.group(1))
            c = float(m.group(2))
            if c == 0:
                return None
            return b / c

        # strict float?
        try:
            return float(t2)
        except Exception:
            pass

        # fallback: first number in string
        m = self._re_first_num.search(t2)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
        return None


class TableContext:
    """
    Wrapper around DataFrame:
    - guarantees index 0..n-1
    - resolves columns case-insensitively
    - normalizes text and extracts numbers
    """

    def __init__(self, df: pd.DataFrame, num_parser: NumberParser) -> None:
        self.df = df.reset_index(drop=True)
        self.num_parser = num_parser
        self.col_map = self._build_col_map(self.df)

    def _build_col_map(self, df: pd.DataFrame) -> Dict[str, str]:
        m: Dict[str, str] = {}
        for c in df.columns:
            m[str(c).strip().lower()] = c
        return m

    def resolve_col(self, field_name: str) -> str:
        key = str(field_name).strip().lower()
        if key in self.col_map:
            return self.col_map[key]
        raise EvalError(f"Unknown column: {field_name}")

    def cell(self, row: int, field_name: str) -> str:
        col = self.resolve_col(field_name)
        v = self.df.at[row, col]
        if v is None:
            return ""
        return str(v)

    def norm_text(self, s: Any) -> str:
        # Structural values cannot be meaningfully compared as text.
        # Raising here causes cmp_eq to propagate the error, which is the correct
        # behavior when a query mistakenly passes a multi-row result to eq/not_eq.
        if isinstance(s, (RowSet, ScalarList)):
            raise EvalError(
                f"Cannot compare {type(s).__name__} as text: "
                "use hop{{...}} to extract a single cell value first."
            )
        t = "" if s is None else str(s)
        t = t.strip()

        # Convert HTML quotes to regular quotes
        t = t.replace("&#34;", '"').replace("&quot;", '"')

        # Remove outer quotes if they actually wrap the string
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
            t = t[1:-1].strip()

        return t.strip().lower()

    def to_number(self, s: Any) -> Optional[float]:
        # bool must not be converted to a number (True → 1 / False → 0 would break logic)
        if isinstance(s, bool):
            return None
        if isinstance(s, (int, float)):
            return float(s)
        # RowSet and ScalarList are structural values, not scalars.
        # Converting them via str() would make parse_literal extract the first digit
        # from the dataclass repr ("RowSet(rows=[0, ...])" → 0,
        # "ScalarList(values=['2', ...])" → 2), producing silently wrong comparisons.
        if isinstance(s, (RowSet, ScalarList)):
            return None
        return self.num_parser.parse_literal(str(s))

    def cmp_eq(self, a: Any, b: Any) -> bool:
        na = self.to_number(a)
        nb = self.to_number(b)
        if na is not None and nb is not None:
            return na == nb
        return self.norm_text(a) == self.norm_text(b)

    def cmp_not_eq(self, a: Any, b: Any) -> bool:
        return not self.cmp_eq(a, b)

    def cmp_greater(self, a: Any, b: Any) -> bool:
        na = self.to_number(a)
        nb = self.to_number(b)
        if na is not None and nb is not None:
            return na > nb
        # if there are no numbers, it is better not to do lexicographic comparison and to treat it as invalid
        raise EvalError("greater requires numeric operands")

    def cmp_less(self, a: Any, b: Any) -> bool:
        na = self.to_number(a)
        nb = self.to_number(b)
        if na is not None and nb is not None:
            return na < nb
        raise EvalError("less requires numeric operands")

    def cmp_greater_eq(self, a: Any, b: Any) -> bool:
        na = self.to_number(a)
        nb = self.to_number(b)
        if na is not None and nb is not None:
            return na >= nb
        raise EvalError("greater_eq requires numeric operands")

    def cmp_less_eq(self, a: Any, b: Any) -> bool:
        na = self.to_number(a)
        nb = self.to_number(b)
        if na is not None and nb is not None:
            return na <= nb
        raise EvalError("less_eq requires numeric operands")


class ProgramParser:
    """
    DSL parser:
    - quote-aware + brace-depth-aware
    - supports delimiters ';' and ','
    - performs arity repair for real noise from bootstrap_full.json
    """

    _re_ident = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\{")

    def __init__(self, registry: FunctionRegistry) -> None:
        self.registry = registry
        self._cache: Dict[str, Program] = {}

    def parse_program(self, text: str) -> Program:
        key = text
        if key in self._cache:
            return self._cache[key]

        s = self._preprocess(text.strip())
        eq_pos = self._find_last_top_level_equals(s)
        if eq_pos is None:
            raise ParseError("Program must end with =True or =False")

        expr_str = s[:eq_pos].strip()
        suffix = s[eq_pos + 1:].strip().lower()
        if suffix not in ("true", "false"):
            raise ParseError(f"Bad suffix: {suffix}")

        prog = Program(expr=self.parse_expr(expr_str), expected=(suffix == "true"))
        self._cache[key] = prog
        return prog

    def parse_expr(self, text: str) -> Any:
        t = self._preprocess(text.strip())
        m = self._re_ident.match(t)
        if not m:
            return Literal(t)

        name = m.group(1)
        name_c = self.registry.canonicalize(name)

        # position of the opening '{' immediately after the name
        open_pos = len(name)
        if open_pos >= len(t) or t[open_pos] != "{":
            return Literal(t)

        close_pos = self._find_matching_brace(t, open_pos)
        if close_pos is None or close_pos != len(t) - 1:
            # either it was not closed, or there is trailing content -> treat it as a literal (conservatively)
            return Literal(t)

        inside = t[open_pos + 1: close_pos]
        raw_args = self._split_args(inside)
        repaired = self._repair_args(name_c, raw_args)

        args_ast = [self.parse_expr(a) for a in repaired]
        return Call(name=name_c, args=args_ast)

    def _preprocess(self, s: str) -> str:
        # Important: convert HTML quotes to regular quotes so the splitter can ignore delimiters inside quotes
        return s.replace("&#34;", '"').replace("&quot;", '"')

    def _find_last_top_level_equals(self, s: str) -> Optional[int]:
        depth = 0
        in_quote = False
        pos = None
        for i, ch in enumerate(s):
            if ch == '"':
                in_quote = not in_quote
                continue
            if in_quote:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == "=" and depth == 0:
                pos = i
        return pos

    def _find_matching_brace(self, s: str, open_pos: int) -> Optional[int]:
        depth = 0
        in_quote = False
        for i in range(open_pos, len(s)):
            ch = s[i]
            if ch == '"':
                in_quote = not in_quote
                continue
            if in_quote:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        return None

    def _has_top_level_semicolon(self, s: str) -> bool:
        depth = 0
        in_quote = False
        for ch in s:
            if ch == '"':
                in_quote = not in_quote
                continue
            if in_quote:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == ";" and depth == 0:
                return True
        return False

    def _split_args(self, inside: str) -> List[str]:
        """
        Splits arguments inside {...} by the top-level delimiter:
        - if there is ';' at depth=0 => split by ';'
        - otherwise split by ','
        quote-aware and brace-aware.
        Important: DO NOT discard empty arguments.
        """
        delim = ";" if self._has_top_level_semicolon(inside) else ","

        parts: List[str] = []
        buf: List[str] = []
        depth = 0
        in_quote = False

        for ch in inside:
            if ch == '"':
                in_quote = not in_quote
                buf.append(ch)
                continue

            if not in_quote:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1

                if ch == delim and depth == 0:
                    parts.append("".join(buf).strip())
                    buf = []
                    continue

            buf.append(ch)

        parts.append("".join(buf).strip())
        return parts

    def _repair_args(self, canonical_name: str, raw_args: List[str]) -> List[str]:
        """
        Fixes real noise patterns:
        - filter_* expects 3 args, but 2 arrive: "field, value" in one argument -> split them
        - if there are too many arguments (usually because of commas) -> join the tail into the last one
        - unary ops with extra arguments -> drop the extras (count/only/etc)
        """
        spec = self.registry.get_spec(canonical_name)
        if spec is None:
            return raw_args

        args = list(raw_args)

        # 1) special cases: filter_* with 2 args instead of 3 (a specific noise pattern)
        if canonical_name in ("filter_less", "filter_eq", "filter_not_eq",
                              "filter_greater", "filter_greater_eq", "filter_less_eq") and len(args) == 2:
            # pattern: filter_less{C; field, value}
            split = self._split_field_value_pair(args[1])
            if split is not None:
                args = [args[0], split[0], split[1]]

        # 2) rare noise: filter_eq may arrive with 4 arguments because of commas in value
        #    reduce to 3 by joining the tail
        if canonical_name.startswith("filter_") and spec.max_args == 3 and len(args) > 3:
            args = [args[0], args[1], self._join_tail(args[2:])]

        # 3) comparisons like "not_eq" can have 3 arguments because of commas -> join into 2
        if canonical_name in ("eq", "not_eq", "greater", "less") and len(args) > 2:
            args = [args[0], self._join_tail(args[1:])]

        # 4) unary ops with extra arguments (this appears in local edits)
        if canonical_name in ("count", "only", "top", "bottom", "half", "any", "none", "zero", "not") and len(args) > 1:
            args = [args[0]]

        # 5) add/diff are strictly binary: if there are suddenly 3 -> join into 2
        if canonical_name in ("add", "diff") and len(args) > 2:
            args = [args[0], self._join_tail(args[1:])]

        # 6) all_eq / all_greater allow a 2-arg variant (rare) and a 3-arg variant (usual).
        #    if >3 -> join the tail into the last one
        if canonical_name in ("all_eq", "all_greater") and len(args) > 3:
            args = [args[0], args[1], self._join_tail(args[2:])]

        # 7) and/or — variadic: just trim empty tails, but not below min_args
        #    (empty arguments are sometimes produced by ";;")
        if canonical_name in ("and", "or"):
            args = [a for a in args if a != ""]
            # if after cleanup there are too few arguments, leave it as is (it will fail in Eval)

        # finally: if max_args is set and there are still too many arguments — join the tail
        if spec.max_args is not None and len(args) > spec.max_args:
            head = args[:spec.max_args - 1]
            tail = args[spec.max_args - 1:]
            args = head + [self._join_tail(tail)]

        return args

    def _split_field_value_pair(self, s: str) -> Optional[Tuple[str, str]]:
        """
        Splits the string "field, value" into (field, value), but only at the top level and with quotes taken into account.
        We take the last occurrence of ',' (usually value may contain commas).
        """
        t = s.strip()
        if not t or "," not in t:
            return None

        in_quote = False
        last_comma = None
        for i, ch in enumerate(t):
            if ch == '"':
                in_quote = not in_quote
                continue
            if not in_quote and ch == ",":
                last_comma = i

        if last_comma is None:
            return None

        left = t[:last_comma].strip()
        right = t[last_comma + 1:].strip()
        if not left or not right:
            return None
        return left, right

    def _join_tail(self, parts: List[str]) -> str:
        # Join with "," — this is safer: most often the tail was created by commas inside literals.
        return ", ".join([p.strip() for p in parts])


class TabFactEngine:
    """
    DSL executor:
    - parse -> AST -> eval
    - strict types (RowSet/ScalarList)
    - operations are implemented as methods (without nested functions)
    """

    def __init__(self) -> None:
        self.registry = FunctionRegistry()
        self.parser = ProgramParser(self.registry)
        self.num_parser = NumberParser()

        # dispatch table: canonical_name -> method
        self._ops: Dict[str, Any] = {
            # comparisons
            "eq": self.op_eq,
            "not_eq": self.op_not_eq,
            "greater": self.op_greater,
            "less": self.op_less,

            # boolean logic
            "and": self.op_and,
            "or": self.op_or,
            "not": self.op_not,

            # filters
            "filter_eq": self.op_filter_eq,
            "filter_not_eq": self.op_filter_not_eq,
            "filter_greater": self.op_filter_greater,
            "filter_greater_eq": self.op_filter_greater_eq,
            "filter_less": self.op_filter_less,
            "filter_less_eq": self.op_filter_less_eq,

            # navigation
            "hop": self.op_hop,
            "argmax": self.op_argmax,
            "argmin": self.op_argmin,
            "top": self.op_top,
            "bottom": self.op_bottom,

            # aggregates
            "count": self.op_count,
            "sum": self.op_sum,
            "avg": self.op_avg,
            "max": self.op_max,
            "min": self.op_min,
            "uniq": self.op_uniq,
            "most_freq": self.op_most_freq,
            "half": self.op_half,

            # quantifiers
            "within": self.op_within,
            "not_within": self.op_not_within,
            "any_eq": self.op_within,   # any_eq is semantically the same as within
            "any": self.op_any,
            "none": self.op_none,
            "only": self.op_only,
            "zero": self.op_zero,

            # all_*
            "all_eq": self.op_all_eq,
            "all_not_eq": self.op_all_not_eq,
            "all_greater": self.op_all_greater,
            "all_greater_eq": self.op_all_greater_eq,
            "all_less": self.op_all_less,
            "all_less_eq": self.op_all_less_eq,

            # arithmetic
            "diff": self.op_diff,
            "add": self.op_add,

            # order/positional
            "before": self.op_before,
            "after": self.op_after,
            "first": self.op_first,
            "second": self.op_second,
            "third": self.op_third,
            "fourth": self.op_fourth,
            "fifth": self.op_fifth,
            "last": self.op_last,

            # rare
            "rank": self.op_rank,
        }

    def execute(self, program_text: str, df: pd.DataFrame) -> ExecResult:
        """
        Returns ExecResult:
        - if successful: final != None
        - otherwise: final == None and error is present
        """
        try:
            prog = self.parser.parse_program(program_text)
            ctx = TableContext(df, self.num_parser)
            value = self.eval_node(prog.expr, ctx)

            if not isinstance(value, bool):
                raise EvalError(f"Expression did not evaluate to bool, got: {type(value)}")

            final = (value == prog.expected)
            return ExecResult(final=final, expr_value=value, expected=prog.expected, error=None, executable=True)

        except Exception as e:
            return ExecResult(final=None, expr_value=None, expected=None, error=str(e), executable=False)

    def eval_node(self, node: Any, ctx: TableContext) -> Any:
        if isinstance(node, Literal):
            return self.eval_literal(node, ctx)
        if isinstance(node, Call):
            return self.eval_call(node, ctx)
        raise EvalError(f"Bad AST node: {type(node)}")

    def eval_literal(self, node: Literal, ctx: TableContext) -> Any:
        raw = node.raw.strip()

        # special rowset literal
        if raw == "all_rows":
            return RowSet(rows=list(range(len(ctx.df))), hint_field=None)

        low = raw.lower()
        if low == "true":
            return True
        if low == "false":
            return False

        # keep as string; numbers will be parsed where needed via ctx.to_number(...)
        # but we normalize quotes for literals that are written as "..."
        # (this matters for weird program with "kuala lumpur;")
        if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
            return raw[1:-1]

        return raw

    def eval_call(self, node: Call, ctx: TableContext) -> Any:
        name = self.registry.canonicalize(node.name)
        if name not in self._ops:
            raise EvalError(f"Unknown function: {name}")

        args = [self.eval_node(a, ctx) for a in node.args]
        return self._ops[name](args, ctx)

    def as_rowset(self, v: Any) -> RowSet:
        if isinstance(v, RowSet):
            return v
        raise EvalError(f"Expected RowSet, got: {type(v)}")

    def as_row(self, v: Any) -> int:
        if isinstance(v, int):
            return v
        if isinstance(v, RowSet) and v.size() == 1:
            return v.top()
        raise EvalError(f"Expected Row, got: {type(v)}")

    def as_scalar_list(self, v: Any) -> ScalarList:
        if isinstance(v, ScalarList):
            return v
        raise EvalError(f"Expected ScalarList, got: {type(v)}")

    def row_of(self, v: Any) -> int:
        # For before/after and positional ops: RowSet -> top row
        if isinstance(v, int):
            return v
        if isinstance(v, RowSet):
            return v.top()
        raise EvalError(f"Expected Row or RowSet, got: {type(v)}")

    # Operations: comparisons

    def op_eq(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("eq", args, 2)
        return ctx.cmp_eq(args[0], args[1])

    def op_not_eq(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("not_eq", args, 2)
        return ctx.cmp_not_eq(args[0], args[1])

    def op_greater(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("greater", args, 2)
        return ctx.cmp_greater(args[0], args[1])

    def op_less(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("less", args, 2)
        return ctx.cmp_less(args[0], args[1])

    # Operations: boolean logic

    def op_and(self, args: List[Any], ctx: TableContext) -> bool:
        if len(args) < 2:
            raise EvalError("and requires at least 2 args")
        return all(bool(a) for a in args)

    def op_or(self, args: List[Any], ctx: TableContext) -> bool:
        if len(args) < 2:
            raise EvalError("or requires at least 2 args")
        return any(bool(a) for a in args)

    def op_not(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("not", args, 1)
        return not bool(args[0])

    # Operations: filters

    def op_filter_eq(self, args: List[Any], ctx: TableContext) -> RowSet:
        self._require_arity("filter_eq", args, 3)
        C = self.as_rowset(args[0])
        field = str(args[1])
        value = args[2]
        out = []
        for r in C.rows:
            if ctx.cmp_eq(ctx.cell(r, field), value):
                out.append(r)
        return RowSet(rows=out, hint_field=field)

    def op_filter_not_eq(self, args: List[Any], ctx: TableContext) -> RowSet:
        self._require_arity("filter_not_eq", args, 3)
        C = self.as_rowset(args[0])
        field = str(args[1])
        value = args[2]
        out = []
        for r in C.rows:
            if ctx.cmp_not_eq(ctx.cell(r, field), value):
                out.append(r)
        return RowSet(rows=out, hint_field=field)

    def op_filter_greater(self, args: List[Any], ctx: TableContext) -> RowSet:
        self._require_arity("filter_greater", args, 3)
        C = self.as_rowset(args[0])
        field = str(args[1])
        value = args[2]
        out = []
        for r in C.rows:
            cell_v = ctx.cell(r, field)
            try:
                if ctx.cmp_greater(cell_v, value):
                    out.append(r)
            except EvalError:
                # if it is not numeric, it simply does not pass the filter
                continue
        return RowSet(rows=out, hint_field=field)

    def op_filter_greater_eq(self, args: List[Any], ctx: TableContext) -> RowSet:
        self._require_arity("filter_greater_eq", args, 3)
        C = self.as_rowset(args[0])
        field = str(args[1])
        value = args[2]
        out = []
        for r in C.rows:
            cell_v = ctx.cell(r, field)
            try:
                if ctx.cmp_greater_eq(cell_v, value):
                    out.append(r)
            except EvalError:
                continue
        return RowSet(rows=out, hint_field=field)

    def op_filter_less(self, args: List[Any], ctx: TableContext) -> RowSet:
        self._require_arity("filter_less", args, 3)
        C = self.as_rowset(args[0])
        field = str(args[1])
        value = args[2]
        out = []
        for r in C.rows:
            cell_v = ctx.cell(r, field)
            try:
                if ctx.cmp_less(cell_v, value):
                    out.append(r)
            except EvalError:
                continue
        return RowSet(rows=out, hint_field=field)

    def op_filter_less_eq(self, args: List[Any], ctx: TableContext) -> RowSet:
        self._require_arity("filter_less_eq", args, 3)
        C = self.as_rowset(args[0])
        field = str(args[1])
        value = args[2]
        out = []
        for r in C.rows:
            cell_v = ctx.cell(r, field)
            try:
                if ctx.cmp_less_eq(cell_v, value):
                    out.append(r)
            except EvalError:
                continue
        return RowSet(rows=out, hint_field=field)

    # Operations: navigation

    def op_hop(self, args: List[Any], ctx: TableContext) -> Union[Scalar, ScalarList]:
        self._require_arity("hop", args, 2)
        base = args[0]
        field = str(args[1])

        # RowSet -> scalar if size==1, else ScalarList
        if isinstance(base, RowSet):
            if base.size() == 1:
                return ctx.cell(base.top(), field)
            return ScalarList([ctx.cell(r, field) for r in base.rows])

        # Row -> scalar
        if isinstance(base, int):
            return ctx.cell(base, field)

        raise EvalError(f"hop expects Row or RowSet, got: {type(base)}")

    def op_argmax(self, args: List[Any], ctx: TableContext) -> int:
        if len(args) == 1:
            C = self.as_rowset(args[0])
            if C.hint_field is None:
                raise EvalError("argmax{C} requires hint_field (use argmax{C; field})")
            field = C.hint_field
        else:
            self._require_arity("argmax", args, 2)
            C = self.as_rowset(args[0])
            field = str(args[1])

        best_r = None
        best_v = None
        for r in C.rows:
            n = ctx.to_number(ctx.cell(r, field))
            if n is None:
                continue
            if best_r is None or n > best_v:
                best_r = r
                best_v = n
        if best_r is None:
            raise EvalError("argmax: no numeric values")
        return best_r

    def op_argmin(self, args: List[Any], ctx: TableContext) -> int:
        if len(args) == 1:
            C = self.as_rowset(args[0])
            if C.hint_field is None:
                raise EvalError("argmin{C} requires hint_field (use argmin{C; field})")
            field = C.hint_field
        else:
            self._require_arity("argmin", args, 2)
            C = self.as_rowset(args[0])
            field = str(args[1])

        best_r = None
        best_v = None
        for r in C.rows:
            n = ctx.to_number(ctx.cell(r, field))
            if n is None:
                continue
            if best_r is None or n < best_v:
                best_r = r
                best_v = n
        if best_r is None:
            raise EvalError("argmin: no numeric values")
        return best_r

    def op_top(self, args: List[Any], ctx: TableContext) -> int:
        self._require_arity("top", args, 1)
        C = self.as_rowset(args[0])
        return C.top()

    def op_bottom(self, args: List[Any], ctx: TableContext) -> int:
        self._require_arity("bottom", args, 1)
        C = self.as_rowset(args[0])
        return C.bottom()

    # Operations: aggregates

    def op_count(self, args: List[Any], ctx: TableContext) -> float:
        self._require_arity("count", args, 1)
        C = self.as_rowset(args[0])
        return float(C.size())

    def op_sum(self, args: List[Any], ctx: TableContext) -> float:
        if len(args) == 1:
            xs = self._values_from_scalarlist_arg(args[0])
            return float(sum(xs))
        self._require_arity("sum", args, 2)
        xs = self._values_from_rowset_field(args[0], args[1], ctx)
        return float(sum(xs))

    def op_avg(self, args: List[Any], ctx: TableContext) -> float:
        if len(args) == 1:
            xs = self._values_from_scalarlist_arg(args[0])
        else:
            self._require_arity("avg", args, 2)
            xs = self._values_from_rowset_field(args[0], args[1], ctx)

        if not xs:
            raise EvalError("avg: empty/non-numeric")
        return float(sum(xs) / len(xs))

    def op_max(self, args: List[Any], ctx: TableContext) -> float:
        self._require_arity("max", args, 2)
        xs = self._values_from_rowset_field(args[0], args[1], ctx)
        if not xs:
            raise EvalError("max: empty/non-numeric")
        return float(max(xs))

    def op_min(self, args: List[Any], ctx: TableContext) -> float:
        self._require_arity("min", args, 2)
        xs = self._values_from_rowset_field(args[0], args[1], ctx)
        if not xs:
            raise EvalError("min: empty/non-numeric")
        return float(min(xs))

    def op_uniq(self, args: List[Any], ctx: TableContext) -> float:
        self._require_arity("uniq", args, 2)
        C = self.as_rowset(args[0])
        field = str(args[1])
        seen = set()
        for r in C.rows:
            seen.add(ctx.norm_text(ctx.cell(r, field)))
        return float(len(seen))

    def op_most_freq(self, args: List[Any], ctx: TableContext) -> str:
        self._require_arity("most_freq", args, 2)
        C = self.as_rowset(args[0])
        field = str(args[1])
        counts: Dict[str, int] = {}
        for r in C.rows:
            key = ctx.norm_text(ctx.cell(r, field))
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            raise EvalError("most_freq: empty")
        # max by count, tie-break lexicographically for determinism
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return best

    def op_half(self, args: List[Any], ctx: TableContext) -> float:
        self._require_arity("half", args, 1)
        C = self.as_rowset(args[0])
        return float(C.size()) / 2.0

    # Operations: quantifiers

    def op_within(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("within", args, 3)
        C = self.as_rowset(args[0])
        field = str(args[1])
        value = args[2]
        for r in C.rows:
            if ctx.cmp_eq(ctx.cell(r, field), value):
                return True
        return False

    def op_not_within(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("not_within", args, 3)
        return not self.op_within(args, ctx)

    def op_any(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("any", args, 1)
        v = args[0]
        if isinstance(v, RowSet):
            return not v.is_empty()
        if isinstance(v, ScalarList):
            return not v.is_empty()
        if v is None:
            return False
        if isinstance(v, str):
            return bool(v.strip())
        return True

    def op_none(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("none", args, 1)
        return not self.op_any(args, ctx)

    def op_only(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("only", args, 1)
        C = self.as_rowset(args[0])
        return C.size() == 1

    def op_zero(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("zero", args, 1)
        n = ctx.to_number(args[0])
        if n is None:
            raise EvalError("zero requires numeric operand")
        return n == 0.0

    # Operations: all_*

    def op_all_eq(self, args: List[Any], ctx: TableContext) -> bool:
        if len(args) == 2:
            # rare form: all_eq{ScalarList; Value} (there are actually 3 such cases in the dataset)
            lst = self.as_scalar_list(args[0])
            val = args[1]
            for x in lst.values:
                if not ctx.cmp_eq(x, val):
                    return False
            return True

        self._require_arity("all_eq", args, 3)
        C = self.as_rowset(args[0])
        field = str(args[1])
        val = args[2]
        for r in C.rows:
            if not ctx.cmp_eq(ctx.cell(r, field), val):
                return False
        return True

    def op_all_not_eq(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("all_not_eq", args, 3)
        C = self.as_rowset(args[0])
        field = str(args[1])
        val = args[2]
        for r in C.rows:
            if ctx.cmp_eq(ctx.cell(r, field), val):
                return False
        return True

    def op_all_greater(self, args: List[Any], ctx: TableContext) -> bool:
        if len(args) == 2:
            # rare form: all_greater{ScalarList; Value} (there are 2 such cases in the dataset)
            lst = self.as_scalar_list(args[0])
            val = args[1]
            for x in lst.values:
                if not ctx.cmp_greater(x, val):
                    return False
            return True

        self._require_arity("all_greater", args, 3)
        C = self.as_rowset(args[0])
        field = str(args[1])
        val = args[2]
        for r in C.rows:
            if not ctx.cmp_greater(ctx.cell(r, field), val):
                return False
        return True

    def op_all_greater_eq(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("all_greater_eq", args, 3)
        C = self.as_rowset(args[0])
        field = str(args[1])
        val = args[2]
        for r in C.rows:
            if not ctx.cmp_greater_eq(ctx.cell(r, field), val):
                return False
        return True

    def op_all_less(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("all_less", args, 3)
        C = self.as_rowset(args[0])
        field = str(args[1])
        val = args[2]
        for r in C.rows:
            if not ctx.cmp_less(ctx.cell(r, field), val):
                return False
        return True

    def op_all_less_eq(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("all_less_eq", args, 3)
        C = self.as_rowset(args[0])
        field = str(args[1])
        val = args[2]
        for r in C.rows:
            if not ctx.cmp_less_eq(ctx.cell(r, field), val):
                return False
        return True

    # Operations: arithmetic

    def op_diff(self, args: List[Any], ctx: TableContext) -> float:
        self._require_arity("diff", args, 2)
        a = ctx.to_number(args[0])
        b = ctx.to_number(args[1])
        if a is None or b is None:
            raise EvalError("diff requires numeric operands")
        return float(a - b)

    def op_add(self, args: List[Any], ctx: TableContext) -> float:
        self._require_arity("add", args, 2)
        a = ctx.to_number(args[0])
        b = ctx.to_number(args[1])
        if a is None or b is None:
            raise EvalError("add requires numeric operands")
        return float(a + b)

    # Operations: order/positional

    def op_before(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("before", args, 2)
        ra = self.row_of(args[0])
        rb = self.row_of(args[1])
        return ra < rb

    def op_after(self, args: List[Any], ctx: TableContext) -> bool:
        self._require_arity("after", args, 2)
        ra = self.row_of(args[0])
        rb = self.row_of(args[1])
        return ra > rb

    def op_first(self, args: List[Any], ctx: TableContext) -> bool:
        return self._positional(args, ctx, pos=1)

    def op_second(self, args: List[Any], ctx: TableContext) -> bool:
        return self._positional(args, ctx, pos=2)

    def op_third(self, args: List[Any], ctx: TableContext) -> bool:
        return self._positional(args, ctx, pos=3)

    def op_fourth(self, args: List[Any], ctx: TableContext) -> bool:
        return self._positional(args, ctx, pos=4)

    def op_fifth(self, args: List[Any], ctx: TableContext) -> bool:
        return self._positional(args, ctx, pos=5)

    def op_last(self, args: List[Any], ctx: TableContext) -> bool:
        # last{C; D}: D is in the last position in C
        self._require_arity("last", args, 2)
        C = self.as_rowset(args[0])
        drow = self.row_of(args[1])
        if C.is_empty():
            return False
        return C.bottom() == drow

    def _positional(self, args: List[Any], ctx: TableContext, pos: int) -> bool:
        self._require_arity("first/second/..", args, 2)
        C = self.as_rowset(args[0])
        drow = self.row_of(args[1])
        idx = pos - 1
        if C.size() <= idx:
            return False
        return C.rows[idx] == drow

    # Operations: rank (rare)

    def op_rank(self, args: List[Any], ctx: TableContext) -> Scalar:
        """
        rank{C; Field}:
        - if there is a "rank" column in the table: take the row with the minimum rank and hop to Field
        - otherwise: fallback hop{top{C}; Field}
        """
        self._require_arity("rank", args, 2)
        C = self.as_rowset(args[0])
        out_field = str(args[1])

        # if there is no rank column — fallback
        if "rank" not in ctx.col_map:
            r = C.top()
            return ctx.cell(r, out_field)

        best_r = None
        best_v = None
        for r in C.rows:
            n = ctx.to_number(ctx.cell(r, "rank"))
            if n is None:
                continue
            if best_r is None or n < best_v:
                best_r = r
                best_v = n

        if best_r is None:
            best_r = C.top()
        return ctx.cell(best_r, out_field)

    # Helpers: aggregates input

    def _values_from_rowset_field(self, rowset_val: Any, field_val: Any, ctx: TableContext) -> List[float]:
        C = self.as_rowset(rowset_val)
        field = str(field_val)
        xs: List[float] = []
        for r in C.rows:
            n = ctx.to_number(ctx.cell(r, field))
            if n is not None:
                xs.append(n)
        return xs

    def _values_from_scalarlist_arg(self, arg: Any) -> List[float]:
        lst = self.as_scalar_list(arg)
        xs: List[float] = []
        for v in lst.values:
            if isinstance(v, bool):
                continue
            try:
                n = float(str(v).replace(",", ""))
                xs.append(n)
            except Exception:
                continue
        return xs

    # Helpers: arity checks

    def _require_arity(self, name: str, args: List[Any], expected: int) -> None:
        if len(args) != expected:
            raise EvalError(f"{name} expects {expected} args, got {len(args)}")
