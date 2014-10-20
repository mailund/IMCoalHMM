'''
Created on Oct 20, 2014

@author: Paula
'''
import newick.tree, newick


class PairTMRCA(newick.tree.TreeVisitor):
    '''class for finding the TMRCA for all the pairs of leaves in a tree'''
        
    def pre_visit_edge(self, src, bootstrap, length, dst):
        subtree_leaves = set(map(int,dst.get_leaves_identifiers()))
        rest_leaves = self.all_leaves.difference(subtree_leaves)
        # for all leaves that have not been visited already, 
        # such that i is in subtree_leaves and j is in rest_leaves
        # update the corresponding tmrca with length
        for i in subtree_leaves:
            for j in rest_leaves:
                    min_leaf = min(i, j)
                    max_leaf = max(i, j)
                    self.tmrca[(min_leaf, max_leaf)] += length/2

    def get_TMRCA(self, string_tree):
        tree = newick.tree.parse_tree(string_tree)
        self.all_leaves = set(map(int,tree.get_leaves_identifiers()))
        no_leaves = len(self.all_leaves)
        self.tmrca = {}
        # initialize the tmrca to 0
        for i in range(no_leaves-1):
            for j in range(i+1,no_leaves):
                self.tmrca[(i+1,j+1)] = 0
        tree.dfs_traverse(self)
        return self.tmrca

def process_tree(line):
    if line[0] != '[':
        return None, None
    s = line.strip().split("[")[1].split("]")
    return int(s[0]), s[1]

def count_tmrca(subs=4*25*20000*1e-9, filename='forest.nwk', align3=True):
    """
    This function returns a list of coalescences between 
    """
    visitor = PairTMRCA()
    f = open(filename)
    one_two = []
    three_four = []
    one_three = []
    counts=[]
    for line in f:
        count, tree = process_tree(line)
        tmrca = visitor.get_TMRCA(tree)
        one_two.append(tmrca[(1,2)]*subs)
        if align3:
            three_four.append(tmrca[(3,4)]*subs)
            one_three.append(tmrca[(1,3)]*subs)
        counts.append(count)
    f.close()
    if align3:
        return one_two, three_four, one_three,counts
    else:
        return one_two,counts
    