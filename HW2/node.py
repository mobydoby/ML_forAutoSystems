import numpy as np

class Node:
    def __init__(self, threshold=-1, dimension=-1, left:int = -1, right:int = -1,\
                 group = None, label = None, index = None, loss = None):
        self.threshold = threshold
        self.dimension = dimension
        self.next1 = left
        self.next2 = right

        self.group = group # all the example this node applies to (only leaf nodes have this) 
                           # This is ALWAYS a boolean mask!!!(or None)
        self.label = label # only leaf nodes have a label
        self.index = index # only applies to leaves to have a reference to where it was 
        self.loss = loss # loss for this node

    def __repr__(self):
        return f"\n (thresh:{self.threshold},\n  dimension:{self.dimension},\n  left_index:{self.next1},\n  rigth_index:{self.next2})\n"
    
    def isLeaf(self):
        return (self.threshold==-1 and self.dimension == -1 \
            and self.next1==-1 and self.next2==-1)
    
    def isBranch(self):
        return (self.label == None and\
                self.index == None and\
                self.loss == None)
    
    def getLabel(self):
        if self.label == None:
            raise ValueError("this 'leaf' node doesn't have a label, something is wrong")
        return self.label

    def getGroup(self):
        if self.group == None:
            raise ValueError("this 'leaf' node doesn't have a group, something is wrong")
        return 
    
    def getLoss(self):
        if self.loss == None:
            raise ValueError("this 'leaf' node doesn't have a loss variable, something is wrong")
        return self.loss

    def getNext(self):
        assert (self.next1!=-1 and self.next2 !=-1)
        return self.left, self.right
    
    def convertToBranch(self, thresh:float, dim:int, left_ind, right_ind):
        """
        ***Only used on leaf nodes to turn into branch nodes
        """
        if self.threshold != -1: 
            raise ValueError("Threshold for this node already set (probably not a leaf node)")
        if self.dimension != -1: 
            raise ValueError("Dimension for this node already set (probably not a leaf node)")
        self.threshold = thresh
        self.dimension = dim
        self.next1 = left_ind
        self.next2 = right_ind
        self.label = None
        self.index = None
        self.loss = None

    def split(self, features, t = None, d = None):
        """
        returns split data boolean index mask 
        features: 45 x N
        """
        if t == None: t = self.threshold
        if d == None: d = self.dimension

        target_dimension = features[d]
        left_mask = (np.where(target_dimension < t, True, False))
        right_mask = (np.where(target_dimension >= t, True, False))

        valid_left = self.group * left_mask
        valid_right = self.group * right_mask
        return valid_left, valid_right

if __name__ == '__main__':
    N = Node(0.5, 1, 1, 2)
    N2 = Node(0.5, 20, 1, 2)
    A = np.array([[0,1,2,3,4,5,6],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]])
    N.split(A)
    print([N, N2])