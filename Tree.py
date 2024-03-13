# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 13:50:12 2023

@author: benmu
"""

# %% Tree

class Tree:
    
    def __init__(self, xdim, xval, parent=None):
        self.left = None
        self.right = None
        self.parent = parent
        self.xdim = xdim
        self.xval = xval
        if self.parent is None:
            self.depth = 1.
        else:
            self.depth = self.parent.depth + 1.
        self.leftGP = None
        self.rightGP = None
                
    def rotate(self, swap_loc, loc):
        """
        self is above
        swap_loc is below
        loc binary 0 left, 1 right 
        """
        #finding below tree to rotate
        if bool(swap_loc):
            tree = self.right
        else:
            tree = self.left
        
        # swapping parents children
        if self.parent is not None:
            if self.parent.right is self:
                self.parent.right = tree
            else:
                self.parent.left = tree
                
        if bool(swap_loc):
            self.right = None
        else:
            self.left = None
            
        tree.parent = self.parent
        self.parent = tree
        
        if bool(loc):
            tree.right = self
        else:
            tree.left = self
        
        self.rotate_update_depth(tree, loc)
            
            
    def rotate_update_depth(self, rot_tree, loc):
        leaves = [self]
        while leaves:
            temp_leaves = []
            for tree in leaves:
                tree.depth += 1
                if tree.right is not None:
                    temp_leaves.append(tree.right)
                if tree.left is not None:
                    temp_leaves.append(tree.left)
            leaves = temp_leaves
        
        rot_tree.depth -= 1
        
        if bool(loc):
            if rot_tree.left is not None:
                leaves = [rot_tree.left]
            else:
                leaves = []
        else:
            if rot_tree.right is not None:
                leaves = [rot_tree.right]
            else:
                leaves = []
            
        while leaves:
            temp_leaves = []
            for tree in leaves:
                tree.depth -= 1
                if tree.right is not None:
                    temp_leaves.append(tree.right)
                if tree.left is not None:
                    temp_leaves.append(tree.left)
            leaves = temp_leaves
                    
        
    def grow(self, xdim, xval, loc):
        tree = Tree(xdim, xval, self)
        if bool(loc):
            self.right = tree
        else:
            self.left = tree
        return tree
            
    
    def prune(self, loc):
        if bool(loc):
            self.right.parent = None
            self.right = None
        else:
            self.left.parent = None
            self.left = None
            
    def __str__(self):
        return self.print_string()
    
    def print_string(self, rescale_min=None, rescale_max=None):
        if self.parent is None:
            tree_parent = self
        else:
            tree_parent = self
            while tree_parent.parent is not None:
                tree_parent = tree_parent.parent
            
        max_depth = self.tree_max_depth(tree_parent)
        tree_structure = [[] for i in range(max_depth)]
        tree_structure[0].append(tree_parent)
        for i in range(1, max_depth):
            trees = tree_structure[i - 1]
            for j in range(2 ** (i-1)):
                if trees[j] is not None:
                    tree_structure[i].append(trees[j].left)
                    tree_structure[i].append(trees[j].right)#
                else:
                    for k in range(2):
                        tree_structure[i].append(None)      
        
        max_char = 5 ** max_depth
        string = ""     
        
        def rescale(x, minval, maxval):
            return float(x * (maxval - minval) + minval)
        
        for i in range(max_depth):
            i -= 2
            treesign_length = (2 ** (max_depth - i) - 1)
            
            beginswhitespace = " " * int(sum([(2 ** (j - 1)) for j in range(max_depth - i)]) + (treesign_length - 1) / 2  + 1 - 4)
            midwhitespace = " " * (treesign_length* 2 + - 7)
            correcting_space = " " * (treesign_length + 2)
            string += beginswhitespace
            for tree in tree_structure[i + 2]:
                if tree is None:
                    string += correcting_space + midwhitespace
                else:
                    xdim = tree.xdim; xval = tree.xval
                    if rescale_min is not None:
                        xval = rescale(xval, rescale_min[xdim], rescale_max[xdim])
                    if xval < 0:
                        if abs(xval) // 10 > 0:
                            if abs(xval) // 100 > 0:
                                if abs(xval) // 1000 > 0:
                                    string += "(" + str(xdim) + "," + str(round(xval, 4)) + ")" + midwhitespace
                                else:
                                    string += "(" + str(xdim) + "," + str(round(xval, 3)) + ".)" + midwhitespace
                            else:
                                string += "(" + str(xdim) + "," + str(format(xval, f".{1}f")) + ")" + midwhitespace
                        else:
                            string += "(" + str(xdim) + "," + str(format(xval, f".{2}f")) + ")" + midwhitespace
                    else:
                        if abs(xval) // 10 > 0:
                            if abs(xval) // 100 > 0:
                                if abs(xval) // 1000 > 0:
                                    string += "(" + str(xdim) + ", " + str(round(xval, 4)) + ")" + midwhitespace
                                else:
                                    string += "(" + str(xdim) + ", " + str(round(xval, 3)) + ".)" + midwhitespace
                            else:
                                string += "(" + str(xdim) + ", " + str(format(xval, f".{1}f")) + ")" + midwhitespace
                        else:
                            string += "(" + str(xdim) + ", " + str(format(xval, f".{2}f")) + ")" + midwhitespace
            string += "\n"
            
            
            lineseg = "¯" * treesign_length
            beginswhitespace = " " * int(sum([(2 ** (j - 1)) for j in range(max_depth - i)]))
            midwhitespace = " " * (treesign_length)
            correcting_space = " " * (treesign_length + 2)
            string += beginswhitespace
            for tree in tree_structure[i + 2]:
                if tree is None:
                    string += correcting_space + midwhitespace
                else:
                    string += "/" + lineseg + "\\" + midwhitespace
                    
            string += "\n"
            
        return string
    
    def print_rescaled(self, rescale_min, rescale_max):
        print(self.print_string(rescale_min, rescale_max))
            
    def tree_max_depth(self, tree):
        if self.parent is None:
            tree_parent = tree
        else:
            tree_parent = self
            while tree_parent.parent is not None:
                tree_parent = tree_parent.parent
        
        max_depth = 1
        leaves = [tree_parent]
        while leaves:
            temp_leaves = []
            for tree in leaves:
                if tree.left is not None:
                    temp_leaves.append(tree.left)
                    depth = self.calc_depth(tree.left)
                    if depth > max_depth:
                        max_depth = depth
                if tree.right is not None:
                    temp_leaves.append(tree.right)
                    depth = self.calc_depth(tree.right)
                    if depth > max_depth:
                        max_depth = depth
            
            leaves = temp_leaves
                        
        return max_depth
        
    def calc_depth(self, tree):
        tree_parent = tree
        depth = 1
        while tree_parent.parent is not None:
            tree_parent = tree_parent.parent
            depth += 1
        return depth
    
    def num_empty(self):
        if self.parent is None:
            tree_parent = self
        else:
            tree_parent = self
            while tree_parent.parent is not None:
                tree_parent = tree_parent.parent
                
        num_empty = 0
        trees = [tree_parent]
        while trees:
            temp_trees = []
            for tree in trees:
                if tree.right is None:
                    num_empty += 1
                else:
                    temp_trees.append(tree.right)
                
                if tree.left is None:
                    num_empty += 1
                else:
                    temp_trees.append(tree.left)
            
            trees = temp_trees
        
        return num_empty
    
    def num_pruneable(self):
        if self.parent is None:
            tree_parent = self
        else:
            tree_parent = self
            while tree_parent.parent is not None:
                tree_parent = tree_parent.parent
                
        num_pruneable = 0
        trees = [tree_parent]
        while trees:
            temp_trees = []
            for tree in trees:
                if tree.right is not None:
                    temp_trees.append(tree.right)
                
                if tree.left is not None:
                    temp_trees.append(tree.left)
                    
                if tree.left is None and tree.right is None:
                    num_pruneable += 1
            
            trees = temp_trees
        
        return num_pruneable
    
    def get_internals_and_leaves(self):
        internals = []
        leaves = []
        trees = [self]
        while trees:
            temp_trees = []
            for tree in trees:
                if tree.right is None and tree.left is None:
                    leaves.append(tree)
                elif tree.right is None and tree.left is not None:
                    temp_trees.append(tree.left)
                    internals.append(tree)
                elif tree.right is not None and tree.left is None:
                    temp_trees.append(tree.right)
                    internals.append(tree)
                else:
                    temp_trees.append(tree.right)
                    temp_trees.append(tree.left)
                    internals.append(tree)
                    
            trees = temp_trees
        return internals, leaves
                    
    
    def minus_one_GPindex_greater_than(self, loc):
        if self.parent is None:
            tree_parent = self
        else:
            tree_parent = self
            while tree_parent.parent is not None:
                tree_parent = tree_parent.parent
                
        if bool(loc):
            GPloc = self.rightGP
        else:
            GPloc = self.leftGP
        
        trees = [tree_parent]
        greater_than = []
        while trees:
            temp_trees = []
            for tree in trees:
                if tree.right is not None:
                    temp_trees.append(tree.right)
                if tree.rightGP is not None:
                    if tree.rightGP > GPloc:
                        tree.rightGP -= 1
                if tree.left is not None:
                    temp_trees.append(tree.left)
                if tree.leftGP is not None:
                    if tree.leftGP > GPloc:
                        tree.leftGP -= 1
                        
            trees = temp_trees      
            
    def print_GPs(self):
        if self.parent is None:
            tree_parent = self
        else:
            tree_parent = self
            while tree_parent.parent is not None:
                tree_parent = tree_parent.parent
        GP_index = []
        trees = [tree_parent]
        while trees:
            temp_trees = []
            for tree in trees:
                if tree.right is not None:
                    temp_trees.append(tree.right)
                if tree.rightGP is not None:
                    GP_index.append(tree.rightGP)
                if tree.left is not None:
                    temp_trees.append(tree.left)
                if tree.leftGP is not None:
                    GP_index.append(tree.leftGP)
                    
            trees = temp_trees
        return GP_index
            
    
    def get_parent_splits(self, max_n_dim):
        splits = [[] for _ in range(max_n_dim)]
        tree = self
        while tree.parent is not None:
            if tree.parent.left is tree:
                splits[tree.parent.xdim].append([tree.parent.xval, 0])
            else:
                splits[tree.parent.xdim].append([tree.parent.xval, 1])
            tree = tree.parent
        return splits
    
    def get_splits(self):
        if self.parent is None:
            tree_parent = self
        else:
            tree_parent = self
            while tree_parent.parent is not None:
                tree_parent = tree_parent.parent
        trees = [tree_parent]
        locs = []
        while len(trees) > 0:
            temp_trees = []
            for tree in trees:
                locs.append((tree.xdim, tree.xval))
                if tree.right is not None:
                    temp_trees.append(tree.right)
                if tree.left is not None:
                    temp_trees.append(tree.left)
            trees = temp_trees
        return locs
                

if __name__ == "__main__":
    #############################################
    print("Tree plot example:")
    splits = Tree(0, 0.5)
    splits.right = Tree(1,1, splits)
    splits.left = Tree(0, 0.2, splits)
    splits.left.left = Tree(1, 1, splits.left)
    splits.left.right = Tree(1, 1, splits.left)

    print(splits)
    
    ############################################
    print("Rotate example:")
    tr = Tree(0,0)
    tree = Tree(0,1, tr)
    tr.right = tree
    ltree = Tree(1,1, tree)
    tree.left = ltree
    print(tree)
    tr.rotate(1,1)
    print(tree)
    
    ############################################
    print("Grow example:")
    tree = Tree(0,1)
    tree1 = Tree(0,1,tree)
    tree.right = tree1
    print(tree)
    tree2 = tree.grow(0,0,0)
    print(tree)
    tree2.grow(0,0,0)
    print(tree)
    
    ############################################
    print("Prune example:")
    tree = Tree(0,-1)
    tree1 = Tree(0,100,tree)
    tree.right = tree1
    tree2 = Tree(0,1000,tree)
    tree.left = tree2
    print(tree)
    tree2 = tree.prune(1)
    print(tree)
            