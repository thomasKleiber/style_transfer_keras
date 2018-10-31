import json
import numpy as np

class node():
    def __init__(self, name, parent_name, dist, metric):
        self.name = name
        self.parent_name = parent_name
        self.dist = dist
        self.metric = metric
        self.parent = None
        self.childs = None

    def childs_names(self):
        ret = []
        for c in self.childs:
            ret = ret + [c.name]
        return ret

    def name_sanitized(self):
        if self.name.count('/'):
            return '_' + self.name.split('/')[-1].split('.')[-2]
        return '_' + self.name

class tree():

    def __init__(self):
        self.top = None
        self.nodes = {}
        self.new_element_cnt = 0

    def insert(self, name, parent, dist=0, metric=None):
        self.nodes[name] = node(name, parent, dist, metric)
        self.complete_families()

    def insert_new(self, child, dist=None, metric=None):
        self.new_element_cnt = self.new_element_cnt + 1
        parent = str(self.new_element_cnt)
        self.insert(child, parent, dist, metric)
        return parent

    def _reset(self):
        if self.top is not None:
            del(self.nodes[self.top])
            self.top = None
        for n in self.nodes:
            self.nodes[n].parent = None
            self.nodes[n].childs = None

    def _summit(self):
        for n in self.nodes:
            if not self.nodes[n].parent_name in self.nodes.keys():
                top = self.nodes[n].parent_name
                self.insert(top, '', 0)
                return top

    def _find_parent(self, node):
        for n in self.nodes:
            if self.nodes[n].name == node.parent_name:
                return self.nodes[n]
        return None

    def _find_childs(self, node):
        childs = list()
        for n in self.nodes:
            if self.nodes[n].parent_name == node.name:
                childs.append(self.nodes[n])
        return childs

    def finalize(self):
        self._reset()
        self.top = self._summit()
        for n in self.nodes:
            self.nodes[n].parent = self._find_parent(self.nodes[n])
            self.nodes[n].childs = self._find_childs(self.nodes[n])

    def complete_families(self):
        for n in self.nodes:
            if self.nodes[n].parent == None:
                self.nodes[n].parent = self._find_parent(self.nodes[n])
            if self.nodes[n].childs == None:
                self.nodes[n].childs = self._find_childs(self.nodes[n])

    def to_dict(self):
        tr = {}
        for n in self.nodes:
            if self.nodes[n].name is not self.top:
                if type(self.nodes[n].metric) == np.ndarray:
                    mm = self.nodes[n].metric.tolist()
                else:
                    mm = self.nodes[n].metric
                data = (self.nodes[n].parent_name, self.nodes[n].dist, mm)
                tr[self.nodes[n].name] = data
        return tr

    def from_dict(self, tr):
        self._reset()
        for k in tr.keys():
            self.insert(k, tr[k][0], tr[k][1], np.array(tr[k][2]))
        self.finalize()

    def load_json(self, filename):
        with open(filename, 'r') as f:
            tr = json.load(f)
        self.from_dict(tr)

    def save_json(self, filename):
        tr = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(tr, f)

    def descendents(self, nodename, max_recurse=1e6):
        ret = [nodename]
        if max_recurse > 0:
            for d in self.nodes[nodename].childs:
                ret = ret + self.descendents(d.name, max_recurse - 1)
        return ret

    def ancestors(self, node):
        ret = []
        curr = node
        while self.nodes[curr].parent != None:
            curr = self.nodes[curr].parent.name
            ret = ret + [curr]
        return ret

    def _is_leaf(self, node):
        return len(self.nodes[node].childs) == 0

    def filter_leafs(self, lst):
        ret = []
        for l in lst:
            if self._is_leaf(l):
                ret = ret + [l]
        return ret

    def max_depth_at(self, node):
        if self._is_leaf(node): return 0
        dephts = []
        for c in self.nodes[node].childs:
            dephts = dephts + [1 + self.max_depth_at(c.name)] # 1+ is me
        return max(dephts)


    def nth_parent(self, node, n):
        if n > 1:
            if self.nodes[node].parent is not None:
                return self.nth_parent(self.nodes[node].parent.name, n-1)
        return node

    def common_ancestor(self, n1, n2):
        L1 = self.ancestors(n1)
        L2 = self.ancestors(n2)
        for l in L1:
            if l in L2:
                return l, L1.index(l), L2.index(l)

    def most_distant_cousin(self, node):
        Leafs = self.all_leafs()
        max_dist = 0
        cousin, ancestor = '', ''
        for l in Leafs:
            a, d1, d2 = self.common_ancestor(node, l)
            if min(d1, d2) > max_dist:
                max_dist = min(d1, d2)
                cousin = l
                ancestor = a
        return cousin, ancestor, max_dist

    def get_leafs(self, node, max_recurse=1e6):
        return self.filter_leafs(self.descendents(node, max_recurse))

    def get_descendants(self, name, level, nodes_only=True):
        if level == 0:
            if self._is_leaf(name) and nodes_only : return []
            return [self.nodes[name].name]
        D = []
        for n in self.nodes[name].childs:
            D = D + self.get_descendants(n.name, level-1)
        return D

    def level(self, level):
        return self.get_descendants(self.top, level)

    def all_leafs(self):
        return self.get_leafs(self.top)

    def count_leafs(self, node, max_recurse=1e6):
        return len(self.get_leafs(node, max_recurse))

    def repeat(self, fct, node, n=1e6):
        if n > 1:
            if fct(node) is not None:
                return self.repeat(fct, fct(node), n-1)
        return node

