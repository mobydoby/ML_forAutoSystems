import numpy as np

class Node:
    def __init__(self, threshold=-1, dimension=-1, left:int = -1, right:int = -1,\
                 group = None, label = None, index = None):
        self.threshold = threshold
        self.dimension = dimension
        self.next1 = left
        self.next2 = right
        self.group = group # all the example this node applies to (only leaf nodes have this)
        self.label = label # only leaf nodes have a label
        self.index = index # only applies to leaves to have a reference to where it was 

    def __repr__(self):
        return f"\n (thresh:{self.threshold},\n  dimension:{self.dimension},\n  left_index:{self.next1},\n  rigth_index:{self.next2})\n"
    
    def assign_next(N1, N2):
        next1 = N1
        next2 = N2
    
    def isLeaf(self):
        return (self.label!=None and self.next1==-1 and self.next2==-1)
    
    def getLabel(self):
        if self.label == None:
            raise ValueError("this 'leaf' node doesn't have a label, something is wrong")
        return self.label

    def getGroup(self):
        if self.group == None:
            raise ValueError("this 'leaf' node doesn't have a group, something is wrong")
        return 

    def getNext(self):
        assert (self.next1!=-1 and self.next2 !=-1)
        return self.left, self.right

    def split(self, features, group: np.array):
        """
        returns split data boolean index mask
        features: 45 x N
        """
        target_dimension = features[self.dimension]
        left_mask = (np.where(target_dimension<self.threshold, False, True))
        right_mask = (np.where(target_dimension>=self.threshold, False, True))
        valid_left = group * left_mask
        valid_right = group * right_mask
        return valid_left, valid_right

if __name__ == '__main__':
    N = Node(0.5, 1, 1, 2)
    N2 = Node(0.5, 20, 1, 2)
    A = np.array([[0,1,2,3,4,5,6],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]])
    N.split(A)
    print([N, N2])