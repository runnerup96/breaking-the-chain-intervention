"""
CRUXEval trace utilities (adapted from the original notebook idea).

Functionality:
  - make_trace:               record line-by-line execution trace of a function
  - trace_to_text / parse_*:  serialize/deserialize a trace from a textual form
  - perturb_universal:        apply one of 9 perturbation levels (Clean..Compound)
  - applicable_levels:        which levels are meaningful for a given trace
  - simulate_from_trace:      compute the answer implied by a (possibly perturbed) trace
                              by evaluating the function's `return` expression with the
                              final-step locals
  - canonicalize_trace:       canonical form used for strict equality checks
"""

import ast
import copy
import random
import re
import sys
import textwrap


def make_trace(func, *args, **kwargs):
    """
    Run `func(*args, **kwargs)` while recording per-line locals.
    Returns (trace_log, result).

    trace_log is a list of dicts:
        {"line": <line_no>, "locals": {<name>: <value>}}
    `nl_comment` may be added later by perturbations.
    """
    trace_log = []
    target_file = func.__code__.co_filename
    target_name = func.__code__.co_name

    safe_args = copy.deepcopy(args)

    def tracer(frame, event, _arg):
        if (frame.f_code.co_filename == target_file
                and frame.f_code.co_name == target_name
                and event == "line"):
            safe_locals = {}
            for k, v in frame.f_locals.items():
                try:
                    safe_locals[k] = copy.deepcopy(v)
                except Exception:
                    safe_locals[k] = repr(v)
            trace_log.append({"line": frame.f_lineno, "locals": safe_locals})
        return tracer

    sys.settrace(tracer)
    try:
        result = func(*safe_args, **kwargs)
    finally:
        sys.settrace(None)
    return trace_log, result


def trace_to_text(trace) -> str:
    """Serialize a trace to a readable textual form."""
    lines = []
    for step in trace:
        parts = [f"line {step['line']}:"]
        for k, v in step["locals"].items():
            parts.append(f"  {k} = {repr(v)}")
        lines.append("\n".join(parts))
        if "nl_comment" in step:
            lines.append(f"  # {step['nl_comment']}")
    return "\n".join(lines)


_LINE_HEADER_RE = re.compile(r"^\s*line\s+(?P<num>\d+)\s*:\s*$", re.IGNORECASE)
_LOCAL_RE       = re.compile(r"^\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<val>.+?)\s*$")
_COMMENT_RE     = re.compile(r"^\s+#\s*(?P<comment>.+?)\s*$")


def parse_trace_text(text: str):
    """
    Parse a textual trace back into list[dict].
    Returns None on malformed input (no recognizable line headers).
    Values are kept as the raw repr string (we never eval untrusted reprs here).
    """
    if not text:
        return None
    steps = []
    current = None
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m_header = _LINE_HEADER_RE.match(raw)
        if m_header:
            if current is not None:
                steps.append(current)
            current = {"line": int(m_header.group("num")), "locals": {}}
            continue
        m_cmt = _COMMENT_RE.match(raw)
        if m_cmt and current is not None:
            current["nl_comment"] = m_cmt.group("comment")
            continue
        m_loc = _LOCAL_RE.match(raw)
        if m_loc and current is not None:
            name = m_loc.group("name")
            val_repr = m_loc.group("val")
            try:
                value = ast.literal_eval(val_repr)
            except Exception:
                value = val_repr  # keep the raw string if it cannot be evaluated
            current["locals"][name] = value
            continue
        # ignore unrecognized lines (robustness)
    if current is not None:
        steps.append(current)
    return steps if steps else None


def canonicalize_trace(trace):
    """
    Return a canonical, hashable representation:
      tuple of (line, tuple of sorted (name, repr(value)) pairs, optional NL).
    Use repr() so that lists/dicts/tuples compare by structure.
    """
    if trace is None:
        return None
    out = []
    for step in trace:
        locals_pairs = tuple(
            sorted((k, repr(v)) for k, v in step.get("locals", {}).items())
        )
        nl = step.get("nl_comment")
        out.append((int(step["line"]), locals_pairs, nl))
    return tuple(out)


PERTURBATION_NAMES = {
    0: "Clean",
    1: "SingleValueMutation",
    2: "DoubleValueMutation",
    3: "TypeCoercion",
    4: "StepDrop",
    5: "StepDuplicate",
    6: "StepReorder",
    7: "MisleadingNL",
    8: "Compound",
}

# Local Edits levels (deterministic effect on the answer).
# 7 (NL misleading) and 8 (compound) are kept as separate probes,
# accessible via interventions_to_prompt's extra slots.
LOCAL_EDIT_LEVELS = [1, 2, 3, 4, 5, 6]

NL_COMMENTS = [
    "Note: this loop does not execute for this input.",
    "Note: the condition is always False here.",
    "Note: the variable remains unchanged after this line.",
    "Note: this branch is never reached.",
    "Note: the function returns early before this step.",
]


def _mutate_value(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + random.choice([-3, -2, -1, 1, 2, 3])
    if isinstance(v, float):
        return v * -1.5
    if isinstance(v, str):
        return v[::-1] if v else "CORRUPTED"
    if isinstance(v, list):
        if not v:
            return [0]
        c = list(v)
        c[0] = _mutate_value(c[0])
        return c
    if isinstance(v, dict):
        c = dict(v)
        if c:
            k = next(iter(c))
            c[k] = _mutate_value(c[k])
        return c
    if isinstance(v, tuple):
        return tuple(_mutate_value(x) for x in v)
    return None


def _find_mutable(locals_dict, exclude=("__builtins__",)):
    for k, v in locals_dict.items():
        if k not in exclude and not k.startswith("__"):
            return k, v
    return None, None


def perturb_universal(trace, level: int, seed=None):
    """Return a new trace with perturbation `level` applied (does not mutate input)."""
    if seed is not None:
        random.seed(seed)

    t = copy.deepcopy(trace)
    if not t:
        return t

    if level == 0:
        return t

    if level == 1:
        for step in reversed(t):
            k, v = _find_mutable(step["locals"])
            if k is not None:
                step["locals"][k] = _mutate_value(v)
                break
        return t

    if level == 2:
        mutated = 0
        for step in reversed(t):
            k, v = _find_mutable(step["locals"])
            if k is not None:
                step["locals"][k] = _mutate_value(v)
                mutated += 1
            if mutated == 2:
                break
        return t

    if level == 3:
        for step in reversed(t):
            for k, v in step["locals"].items():
                if k.startswith("__"):
                    continue
                if isinstance(v, int) and not isinstance(v, bool):
                    step["locals"][k] = [v] * 2
                    return t
                if isinstance(v, str):
                    step["locals"][k] = len(v)
                    return t
                if isinstance(v, list) and v:
                    step["locals"][k] = str(v[0])
                    return t
        return t

    if level == 4:
        seen = set()
        new_t = []
        removed = False
        for step in t:
            new_keys = set(step["locals"].keys()) - seen
            seen.update(step["locals"].keys())
            if not removed and new_keys:
                removed = True
                continue
            new_t.append(step)
        return new_t if new_t else t

    if level == 5:
        if len(t) >= 2:
            mid = len(t) // 2
            t.insert(mid, copy.deepcopy(t[mid]))
        return t

    if level == 6:
        if len(t) >= 2:
            i = random.randint(0, len(t) - 2)
            t[i], t[i + 1] = t[i + 1], t[i]
        return t

    if level == 7:
        if t:
            i = random.randint(0, len(t) - 1)
            t[i]["nl_comment"] = random.choice(NL_COMMENTS)
        return t

    if level == 8:
        t = perturb_universal(t, level=1, seed=seed)
        t = perturb_universal(t, level=7, seed=seed)
        return t

    raise ValueError(f"Unknown perturbation level: {level}")


def applicable_levels(trace):
    """Which perturbation levels make sense for this trace."""
    levels = {0, 7, 8}
    if not trace:
        return sorted(levels)
    if len(trace) >= 1:
        levels.add(1)
        levels.add(2)
    has_typed = any(
        isinstance(v, (int, str, list))
        for s in trace for k, v in s["locals"].items()
        if not k.startswith("__")
    )
    if has_typed:
        levels.add(3)
    if len(trace) > 1:
        levels.add(4)
        levels.add(5)
        levels.add(6)
    return sorted(levels)


def simulate_from_trace(trace, func_source: str, global_ns=None):
    """
    Evaluate the function's `return` expression using the LAST step's locals.

    Returns the resulting value (any Python object) or a string starting
    with "ERROR:" on failure. Comparison with a model answer is done by the
    caller via str() equality (mirrors the notebook semantics).
    """
    if not trace:
        return "ERROR: empty trace"

    final_locals = dict(trace[-1].get("locals", {}))

    try:
        tree = ast.parse(textwrap.dedent(func_source))
    except SyntaxError as e:
        return f"ERROR: invalid source ({e})"

    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            if node.value is None:
                return None
            expr = ast.unparse(node.value)
            try:
                return eval(expr, global_ns or {}, final_locals)
            except Exception as e:
                return f"ERROR: {e}"

    return "ERROR: no return found"
