import re
import random
from typing import Union, List, Dict, Any

# ---------- AST ----------
class Node:
    pass

class Call(Node):
    def __init__(self, name: str, args: List[Node]):
        self.name = name
        self.args = args
    def __repr__(self): return f"Call({self.name!r}, {self.args!r})"

class Lit(Node):
    def __init__(self, text: str):
        self.text = text
    def __repr__(self): return f"Lit({self.text!r})"

# ---------- Parser ----------
# Grammar sketch (informal):
#   program := expr [=True|=False]
#   expr    := name '{' arg (';' arg)* '}' | literal
#   arg     := expr | literal
#   literal := any text with balanced braces/semicolons handled by nesting; stripped
#
# We parse by scanning, respecting nesting depth.

def parse_program(prog: str) -> (Node, str):
    """Parse <expr>[=<bool>] -> (AST, trailing '=True'/'=False' or '')"""
    tail = ''
    m = re.search(r'(=True|=False)\s*$', prog.strip())
    if m:
        tail = m.group(1)
        core = prog[:m.start()].strip()
    else:
        core = prog.strip()
    node = _parse_expr(core)
    return node, tail

def _parse_expr(s: str) -> Node:
    s = s.strip()
    # function?
    m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\{', s)
    if m:
        name = m.group(1)
        inside = s[len(name)+1:-1]  # drop name and outer {}
        args = _split_top_args(inside)
        return Call(name, [_parse_expr(a) for a in args])
    # otherwise literal
    return Lit(s)

def _split_top_args(s: str) -> List[str]:
    args, buf, depth = [], [], 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == '{':
            depth += 1
            buf.append(c)
        elif c == '}':
            depth -= 1
            buf.append(c)
        elif c == ';' and depth == 0:
            arg = ''.join(buf).strip()
            if arg:
                args.append(arg)
            buf = []
        else:
            buf.append(c)
        i += 1
    last = ''.join(buf).strip()
    if last:
        args.append(last)
    return args

# ---------- Serializer ----------
# def to_str(node: Node) -> str:
#     if isinstance(node, Lit):
#         return node.text
#     if isinstance(node, Call):
#         return f"{node.name}{{" + '; '.join(to_str(a) for a in node.args) + "}}"
#     raise TypeError(node)

def serialize(node: Node, tail: str) -> str:
    return to_str(node) + (tail if tail else '')

def to_str(node: Node) -> str:
    if isinstance(node, Lit):
        return node.text
    if isinstance(node, Call):
        args_str = '; '.join(to_str(a) for a in node.args)
        return f"{node.name}{{{args_str}}}"  # Правильное количество скобок
    raise TypeError(node)

# ---------- Visitors / utilities ----------
# def visit(node: Node, fn):
#     """Preorder traversal; fn(node) may return a replacement node or None."""
#     repl = fn(node)
#     cur = repl if isinstance(repl, Node) else node
#     if isinstance(cur, Call):
#         new_args = []
#         for a in cur.args:
#             new_args.append(visit(a, fn))
#         cur.args = new_args
#     return cur

def visit(node: Node, fn):
    """Preorder traversal; fn(node) may return a replacement node or None."""
    repl = fn(node)
    cur = repl if isinstance(repl, Node) else node  # Используем исходный узел если fn вернул None
    if isinstance(cur, Call):
        new_args = []
        for a in cur.args:
            new_args.append(visit(a, fn))
        cur.args = new_args
    return cur

def replace_first(node: Node, pred, replacer) -> (Node, bool):
    """Replace first node where pred(node) True."""
    changed = False
    def _fn(n):
        nonlocal changed
        if not changed and pred(n):
            changed = True
            return replacer(n)
    new_node = visit(node, _fn)
    return new_node, changed

# ---------- Interventions ----------
def intervene_filter_column(prog: str, from_col: str, to_col: str, seed: int = 0) -> str:
    node, tail = parse_program(prog)
    def pred(n):
        return (isinstance(n, Call) and n.name == 'filter_eq' and
                len(n.args) >= 2 and isinstance(n.args[1], Lit) and n.args[1].text.strip() == from_col)
    def repl(n):
        m = Call(n.name, list(n.args))
        m.args[1] = Lit(to_col)
        return m
    node, ok = replace_first(node, pred, repl)
    return serialize(node, tail)

def intervene_filter_value(prog: str, from_val: str, to_val: str, seed: int = 0) -> str:
    node, tail = parse_program(prog)
    def pred(n):
        return (isinstance(n, Call) and n.name == 'filter_eq' and
                len(n.args) >= 3 and isinstance(n.args[2], Lit) and n.args[2].text.strip() == from_val)
    def repl(n):
        m = Call(n.name, list(n.args))
        m.args[2] = Lit(to_val)
        return m
    node, ok = replace_first(node, pred, repl)
    return serialize(node, tail)

def intervene_eq_constant(prog: str, old_const: str, new_const: str, seed: int = 0) -> str:
    node, tail = parse_program(prog)
    def pred(n):
        return (isinstance(n, Call) and n.name == 'eq' and
                len(n.args) >= 2 and isinstance(n.args[1], Lit) and n.args[1].text.strip() == old_const)
    def repl(n):
        m = Call(n.name, list(n.args))
        m.args[1] = Lit(new_const)
        return m
    node, ok = replace_first(node, pred, repl)
    return serialize(node, tail)

def intervene_hop_target(prog: str, from_col: str, to_col: str, seed: int = 0) -> str:
    node, tail = parse_program(prog)
    def pred(n):
        return (isinstance(n, Call) and n.name == 'hop' and
                len(n.args) >= 2 and isinstance(n.args[1], Lit) and n.args[1].text.strip() == from_col)
    def repl(n):
        m = Call(n.name, list(n.args))
        m.args[1] = Lit(to_col)
        return m
    node, ok = replace_first(node, pred, repl)
    return serialize(node, tail)

def intervene_global_break(prog: str, value_map: Dict[str, str]) -> str:
    """
    Flip EVERY filter_eq value per provided mapping: e.g., {"united states":"canada"}
    Keeps structure identical; should invert truth in many cases.
    """
    node, tail = parse_program(prog)
    
    def fn(n):
        if isinstance(n, Call) and n.name == 'filter_eq' and len(n.args) >= 3 and isinstance(n.args[2], Lit):
            val = n.args[2].text.strip()
            if val in value_map:
                # Создаем копию узла с измененным значением
                new_args = list(n.args)
                new_args[2] = Lit(value_map[val])
                return Call(n.name, new_args)
        # Не возвращаем ничего (None), если замена не требуется
    
    node = visit(node, fn)
    return serialize(node, tail)


def intervene_aggregation_field(prog: str, old_field: str, new_field: str, seed: int = 0) -> str:
    """Change field name in aggregation functions like max, min, sum, avg."""
    node, tail = parse_program(prog)
    def pred(n):
        return (isinstance(n, Call) and n.name in ['max', 'min', 'sum', 'avg', 'count'] and
                len(n.args) >= 1 and isinstance(n.args[0], Lit) and n.args[0].text.strip() == old_field)
    def repl(n):
        m = Call(n.name, list(n.args))
        m.args[0] = Lit(new_field)
        return m
    node, ok = replace_first(node, pred, repl)
    return serialize(node, tail)

def intervene_comparison_constant(prog: str, old_const: str, new_const: str, seed: int = 0) -> str:
    """Change constant in comparison functions like greater, less, etc."""
    node, tail = parse_program(prog)
    def pred(n):
        return (isinstance(n, Call) and n.name in ['greater', 'less', 'greater_eq', 'less_eq'] and
                len(n.args) >= 2 and isinstance(n.args[1], Lit) and n.args[1].text.strip() == old_const)
    def repl(n):
        m = Call(n.name, list(n.args))
        m.args[1] = Lit(new_const)
        return m
    node, ok = replace_first(node, pred, repl)
    return serialize(node, tail)


def intervene_random_semantic_flip(
    prog: str,
    col_distractors: Dict[str, List[str]],
    value_distractors: Dict[str, List[str]],
    entity_swaps: Dict[str, List[str]],
    seed: int = 0,
    num_changes: int = 1
) -> str:
    """
    Apply multiple random semantic changes to ensure diverse interventions.
    Returns a modified program with specified number of changes.
    """
    rng = random.Random(seed)
    current_prog = prog
    changes_applied = 0
    
    for change_index in range(num_changes):
        if changes_applied >= num_changes:
            break
            
        try:
            node, tail = parse_program(current_prog)
        except Exception as e:
            print(f"Parse error: {e}")
            break
        
        intervention_points = []
        
        def collect_intervention_points(n):
            if isinstance(n, Call):
                # For filter functions
                if n.name in ['filter_eq', 'filter_greater', 'filter_not_eq', 'filter_less', 
                             'filter_greater_eq', 'filter_less_eq'] and len(n.args) >= 3:
                    if isinstance(n.args[1], Lit):  # Column name
                        col = n.args[1].text.strip()
                        cand_cols = [x for x in col_distractors.get('filter', []) if x != col]
                        if cand_cols:
                            intervention_points.append(('filter_column', n, 1, cand_cols))
                    
                    if isinstance(n.args[2], Lit):  # Filter value
                        val = n.args[2].text.strip()
                        col_name = n.args[1].text.strip() if isinstance(n.args[1], Lit) else 'value'
                        cand_vals = value_distractors.get(col_name, [])
                        cand_vals = [x for x in cand_vals if x != val]
                        if cand_vals:
                            intervention_points.append(('filter_value', n, 2, cand_vals))
                
                # For hop function
                elif n.name == 'hop' and len(n.args) >= 2 and isinstance(n.args[1], Lit):
                    col = n.args[1].text.strip()
                    cand_cols = [x for x in col_distractors.get('hop', []) if x != col]
                    if cand_cols:
                        intervention_points.append(('hop_target', n, 1, cand_cols))
                
                # For comparison functions
                elif n.name in ['eq', 'not_eq', 'greater', 'less', 'greater_eq', 'less_eq'] and len(n.args) >= 2:
                    for i, arg in enumerate(n.args[1:], 1):
                        if isinstance(arg, Lit):
                            val = arg.text.strip()
                            cand_vals = [x for x in entity_swaps.get('value', []) if x != val]
                            if cand_vals:
                                intervention_points.append((f'{n.name}_constant', n, i, cand_vals))
                
                # For aggregation functions
                elif n.name in ['max', 'min', 'sum', 'avg', 'count'] and len(n.args) >= 1:
                    if isinstance(n.args[0], Lit):
                        val = n.args[0].text.strip()
                        cand_vals = [x for x in col_distractors.get('aggregation', []) if x != val]
                        if cand_vals:
                            intervention_points.append((f'{n.name}_field', n, 0, cand_vals))
        
        visit(node, collect_intervention_points)
        
        if not intervention_points:
            break  # No more changes possible
        
        # Select a random intervention point
        intervention_type, call_node, arg_index, candidates = rng.choice(intervention_points)
        new_value = rng.choice(candidates)
        
        # Apply the intervention
        new_args = list(call_node.args)
        new_args[arg_index] = Lit(new_value)
        new_node = Call(call_node.name, new_args)
        
        # Replace the node in the AST
        def replace_node(old_node):
            if old_node is call_node:
                return new_node
            return old_node
        
        modified_node = visit(node, replace_node)
        current_prog = serialize(modified_node, tail)
        changes_applied += 1
    
    return current_prog

if __name__ == "__main__":
  prog = "and{only{filter_eq{all_rows; nationality; united states}}; eq{hop{filter_eq{all_rows; nationality; united states}; athlete}; shawn crawford}}=True"

  # 1) Change the FILTER VALUE (US → Canada)
  print(intervene_filter_value(prog, from_val="united states", to_val="canada"))
  # -> same structure, but 'united states' becomes 'canada' (should flip truth)

  # 2) Change the EQ CONSTANT (athlete name)
  print(intervene_eq_constant(prog, old_const="shawn crawford", new_const="usain bolt"))

  # 3) Change the HOP TARGET COLUMN (athlete → country)
  print(intervene_hop_target(prog, from_col="athlete", to_col="event"))

  # 4) Change the FILTER COLUMN (nationality → birthplace)
  print(intervene_filter_column(prog, from_col="nationality", to_col="gender"))

  # 5) Global break: flip all values via mapping
  print(intervene_global_break(prog, {"united states": "canada", "shawn crawford":"usain bolt"}))