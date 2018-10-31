from tree_helper import *
from anytree import Node, RenderTree
from anytree.exporter import DotExporter


def _del_anytree_nodes(T, name):
    name_ = T.nodes[name].name_sanitized()
    if name_ in globals():
        exec('del(' + name_ + ')', globals(), globals())
    for n in T.nodes[name].childs:
        _del_anytree_nodes(T, n.name)


def _create_anytree(T, max_depth, no_leafs):
    _del_anytree_nodes(T, T.top)
    _create_anytree_nodes(T, T.top, max_depth, no_leafs)

def _create_anytree_nodes(T, name, max_depth, no_leafs):
    if no_leafs and T._is_leaf(name):
        return
    name_ = T.nodes[name].name_sanitized()
    id_ = name_ + '_#' + str(T.count_leafs(name))
    if T.nodes[name].parent is not None:
        parent_ = T.nodes[name].parent.name_sanitized()
        exec(name_ + '=Node("' + id_ + '", parent=' + parent_ + ')', globals(), globals())
    else:
        exec(name_ + '=Node("' + id_ + '")', globals(), globals())
    if max_depth > 0:
        for n in T.nodes[name].childs:
            _create_anytree_nodes(T, n.name, max_depth-1, no_leafs)

def draw_tree(T, max_depth=1e6, no_leafs=True):
    _create_anytree(T, max_depth, no_leafs)
    top_node = globals().get(T.nodes[T.top].name_sanitized())
    DotExporter(top_node).to_picture("/tmp/tree.png")

def print_tree(T, max_depth=5, no_leafs=True):
    _create_anytree(T, max_depth, no_leafs)
    top_node = globals().get(T.nodes[T.top].name_sanitized())
    for pre, fill, node in RenderTree(top_node):
        print("%s%s" % (pre, node.name))
