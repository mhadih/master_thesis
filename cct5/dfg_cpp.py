"""C++ data-flow graph extraction for CCT5 flow-conditioned embeddings.

Port of the C# walker in the official CCT5 repo
(src/evaluator/CodeBLEU/parser/DFG.py :: DFG_csharp) to tree-sitter-cpp node
types, keeping the exact edge format:
    (code, idx, 'comesFrom' | 'computedFrom', [codes], [idxs])

C# -> C++ node mapping:
    variable_declarator      -> init_declarator (fields: declarator, value)
    bare `int x;`            -> declaration > identifier (no init_declarator)
    postfix_unary_expression -> update_expression (covers ++/-- pre+postfix)
    local_variable_declaration (for-init) -> declaration
    else branch              -> else_clause node (vs 'else' in type name)
    assignment_expression / if_statement / for_statement / while_statement /
    call_expression / return_statement : same names in tree-sitter-cpp.

Requires: pip install tree_sitter tree_sitter_cpp
"""

import re

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

CPP_LANGUAGE = Language(tscpp.language())


def get_parser():
    return Parser(CPP_LANGUAGE)


def remove_comments_cpp(source):
    pattern = re.compile(r'//.*?$|/\*.*?\*/', re.DOTALL | re.MULTILINE)

    def replacer(match):
        s = match.group(0)
        return "\n" if s.startswith("/") and "\n" in s else " "

    temp = []
    for x in re.sub(pattern, replacer, source).split("\n"):
        if x.strip() != "":
            temp.append(x)
    return "\n".join(temp)


def tree_to_token_index(root_node):
    if (len(root_node.children) == 0 or root_node.type in ['string_literal', 'string',
                                                           'character_literal']) and root_node.type != 'comment':
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def tree_to_variable_index(root_node, index_to_code):
    if (len(root_node.children) == 0 or root_node.type in ['string_literal', 'string',
                                                           'character_literal']) and root_node.type != 'comment':
        index = (root_node.start_point, root_node.end_point)
        _, code = index_to_code[index]
        if root_node.type != code:
            return [(root_node.start_point, root_node.end_point)]
        else:
            return []
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_variable_index(child, index_to_code)
        return code_tokens


def index_to_code_token(index, code):
    start_point, end_point = index[0], index[1]
    if start_point[0] == end_point[0]:
        return code[start_point[0]][start_point[1]:end_point[1]]
    s = code[start_point[0]][start_point[1]:]
    for i in range(start_point[0] + 1, end_point[0]):
        s += code[i]
    s += code[end_point[0]][:end_point[1]]
    return s


def _dedup(DFG):
    dic = {}
    for x in DFG:
        if (x[0], x[1], x[2]) not in dic:
            dic[(x[0], x[1], x[2])] = [x[3], x[4]]
        else:
            dic[(x[0], x[1], x[2])][0] = list(set(dic[(x[0], x[1], x[2])][0] + x[3]))
            dic[(x[0], x[1], x[2])][1] = sorted(list(set(dic[(x[0], x[1], x[2])][1] + x[4])))
    return [(x[0], x[1], x[2], y[0], y[1]) for x, y in sorted(dic.items(), key=lambda t: t[0][1])]


def DFG_cpp(root_node, index_to_code, states):
    assignment = ['assignment_expression']
    def_statement = ['init_declarator']
    bare_declaration = ['declaration']
    increment_statement = ['update_expression']
    if_statement = ['if_statement']
    else_clause = ['else_clause']
    for_statement = ['for_statement']
    while_statement = ['while_statement', 'do_statement']
    do_first_statement = []
    states = states.copy()
    if (len(root_node.children) == 0 or root_node.type in ['string_literal', 'string',
                                                           'character_literal']) and root_node.type != 'comment':
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, idx, 'comesFrom', [code], states[code].copy())], states
        else:
            if root_node.type == 'identifier':
                states[code] = [idx]
            return [(code, idx, 'comesFrom', [], [])], states
    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('declarator')
        value = root_node.child_by_field_name('value')
        DFG = []
        if value is None:
            indexs = tree_to_variable_index(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                DFG.append((code, idx, 'comesFrom', [], []))
                states[code] = [idx]
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = DFG_cpp(value, index_to_code, states)
            DFG += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'comesFrom', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in bare_declaration:
        # e.g. `int x;` — no init_declarator child: bare identifiers are definitions
        if any(c.type == 'init_declarator' for c in root_node.children):
            DFG = []
            for child in root_node.children:
                temp, states = DFG_cpp(child, index_to_code, states)
                DFG += temp
            return sorted(DFG, key=lambda x: x[1]), states
        DFG = []
        for child in root_node.children:
            if child.type != 'identifier':
                continue
            idx, code = index_to_code[(child.start_point, child.end_point)]
            DFG.append((code, idx, 'comesFrom', [], []))
            states[code] = [idx]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name('left')
        right_nodes = root_node.child_by_field_name('right')
        DFG = []
        temp, states = DFG_cpp(right_nodes, index_to_code, states)
        DFG += temp
        name_indexs = tree_to_variable_index(left_nodes, index_to_code)
        value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in increment_statement:
        DFG = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in if_statement:
        DFG = []
        current_states = states.copy()
        others_states = []
        flag = False
        for child in root_node.children:
            if child.type in else_clause:
                flag = True
                temp, new_states = DFG_cpp(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
            elif flag is False and child.type not in if_statement:
                temp, current_states = DFG_cpp(child, index_to_code, current_states)
                DFG += temp
            else:
                flag = True
                temp, new_states = DFG_cpp(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if not any(c.type in else_clause for c in root_node.children):
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(DFG, key=lambda x: x[1]), new_states
    elif root_node.type in for_statement:
        DFG = []
        for child in root_node.children:
            temp, states = DFG_cpp(child, index_to_code, states)
            DFG += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = DFG_cpp(child, index_to_code, states)
                DFG += temp
            elif child.type == "declaration":
                flag = True
        return _dedup(DFG), states
    elif root_node.type in while_statement:
        DFG = []
        for _ in range(2):
            for child in root_node.children:
                temp, states = DFG_cpp(child, index_to_code, states)
                DFG += temp
        return _dedup(DFG), states
    else:
        DFG = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = DFG_cpp(child, index_to_code, states)
                DFG += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = DFG_cpp(child, index_to_code, states)
                DFG += temp
        return sorted(DFG, key=lambda x: x[1]), states


def extract_dataflow(code, parser=None):
    """Full student file -> (tokens, dfg_edges, index_to_code). Edges are
    (code, idx, relation, [codes], [idxs]); idx counts tokens from 0."""
    code = remove_comments_cpp(code)
    if parser is None:
        parser = get_parser()
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    lines = code.split("\n")
    code_tokens = [index_to_code_token(x, lines) for x in tokens_index]
    index_to_code = {}
    for idx, (index, tok) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, tok)
    try:
        dfg, _ = DFG_cpp(root_node, index_to_code, {})
    except Exception:
        dfg = []
    return code_tokens, sorted(dfg, key=lambda x: x[1]), index_to_code


def filter_dfg(dfg, index_to_code, scope):
    """Keep edges whose source variable starts inside [scope_start, scope_end)
    (0-based old/new-file line numbers, as in official filter_dfg)."""
    idx_to_pos = {idx: pos[0] for pos, (idx, _) in index_to_code.items()}
    valid = []
    for edge in dfg:
        src_pos = idx_to_pos.get(edge[1])
        if src_pos is None:
            continue
        line = src_pos[0]
        if scope[0] <= line < scope[1]:
            valid.append(edge)
    return valid


def serialize_edges(dfg):
    """Upstream CDG serialization: 'src dst' (comesFrom) / 'dst src'
    (computedFrom), edges joined by <extra_id_0>."""
    parts = []
    for edge in dfg:
        for end_node in edge[3]:
            if edge[2] == "comesFrom":
                parts.append(edge[0] + " " + end_node)
            elif edge[2] == "computedFrom":
                parts.append(end_node + " " + edge[0])
    return "<extra_id_0>".join(parts)
