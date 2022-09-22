import numpy as np
from scipy import stats
from node import Node    
import copy

def calcLabel(labels, group):
    """
    Inputs: Labels: 1 x N array of labels
            group: 1 x N boolean array mask 
                   of the only indices in group
    Returns the most occuring label in within the group
    """
    filtered_labels = labels[group]
    return stats.mode(filtered_labels)

class DecisionTree:

    def __init__(self, depth):
        self.depth = depth
        self.node_list = []
        
        #keeps track of 
        self.leaf_nodes = []
        
        #keeps track of non-null leaf nodes
        self.leaf_count = 0
        
        # global prediction variable to store predictions for inputs
        self.predictions = np.array([])

    def branch(self, X, node_index, group):
        """
        branch is recursive function for calculating predictions.
        The labels variable is updated after every iteration. 
        Node: 
        X: 45 x N feature array
        Y: N x 1 labels
        pred_labels: predictions made by BDT
        """

        decision = self.node_list[node_index]
        #if the decision node is a leaf, we want to label the prediction
        if (decision.isLeaf == True): 
            
            #change the labels of the groups global predictions variable to label of this node
            self.prediction = np.where(group == True, decision.getLabel(), self.prediction)
            
            #store in list outside of tree for easy access when training
            decision_copy = decision
            decision_copy.group = group
            decision_copy.index = node_index
            self.leaf_nodes.append(decision)
       
        # not a leaf node, so we must split 
        else: 
            assert(self.next1!=-1 and self.next2!=-1)
            decision = self.node_list[node_index]

            # left and right group are the "groups" that can be futher split by the next node
            left_group, right_group = decision.split(X)
            (left_index, right_index) = decision.getNext()

            self.branch(X, left_index, left_group)
            self.branch(X, right_index, right_group)

    def predict(self, X):
        """
        Returns: predictions for each feature vector based on regions
        """
        #initialize prediction vector
        self.prediction = np.ones(X.shape[1])
        #start at the head node and call recursive branch function
        self.branch(X, 0)
        return self.prediction    

    def _get_accuracy_after_split(self, X: np.array, Y: np.array, threshold: float, dimension: int, ind: int)->float:
        """
        returns the accuracy after particular split
        Returns: the 0-1 loss of the dimensional threshold. 
            Must iterate through the predictions and compare
            actual label (Y) and the predicted label(majority label per region)
        Inputs: features: 45 x N
                labels: N x 1
        Output: total correct/total_incorrect
        Procedure: all losses and groups are stored in the leaves (leaf nodes)
                   we split the node along the dimension and value. 
                   and calculate the loss for this single split
        """
        leaf = self.leaf_nodes[ind]
        other_loss = sum([i.getLoss() for i in self.leaf_nodes if i != None]) - leaf.getLoss()
        left_mask, right_mask = leaf.split(X, threshold, dimension)
        left_label = stats.mode(Y[left_mask])
        right_label = stats.mode(Y[right_mask])
        loss_for_split = np.sum(np.where(Y[left_mask]!=left_label, 1, 0)) +\
                         np.sum(np.where(Y[right_mask]!=right_label, 1, 0))
        return other_loss + loss_for_split

    def _actual_split(self, node, features, labels):
        """
        Inputs: node - node object to derive 2 leaves from 
                features 45 x N numpy array
                labels - N x 1 array
        Outputs: left and right nodes that were split along dimension specified by the node.
                 with all leaf attributes:
                 group, 
                 label, 
                 loss, 
                 index
        """
        # split the node (returns 2 boolean masks for indicies in left and right group)
        left_group, right_group = node.split(features)

        #create new leaves (and assign labels and loss)
        left_label = calcLabel(labels, left_group)
        right_label = calcLabel(labels, right_group)

        left_loss = np.sum(np.where(labels[left_group] != left_label, 1, 0))
        right_loss = np.sum(np.where(labels[right_group] != right_label, 1, 0))

        left = Node(group=left_group, label=left_label, index = len(self.node_list), loss=left_loss)
        right = Node(group=right_group, label=right_label, index = len(self.node_list)+1, loss=right_loss)

        return left, right

    '''You don't necessarily need the feature_bounds variable.
    You can derive it from the features vector.
    It is used in the threshold exhaustive search algorithm.'''
    def train(self, features, labels, feature_bounds, depth):
        """
        Creates nodes which represent splits. 
        the thresholds are being trained: t
        the best split is the one that produces the best accuracy 
        across all dimesnsions. 
        The 0-1 loss function for each split is considered. 
        """
        # add the head node
        head = Node(label = stats.mode(labels), \
                    group = np.ma.make_mask(np.ones(labels.shape[0])),
                    loss = labels.shape[0])
        self.node_list.append(head) #initialize node list to [head]
        self.leaf_nodes.append(head) #initialize leaf list to [head]
        self.leaf_count = 1
        while (self.leaf_count < depth):
            #keep track of the best split to create node later
            best = {
                "loss":features.shape[1],
                "thresh":0, 
                "dim":0,
                "leaf_index":0
            }
            #get groups/regions from current tree structure 
            #find best dimension globally
            for ind in range(self.leaf_nodes):
                # sinc leaf nodes' indicies are tracked, deleting them from the list is not efficient.
                # the leaf nodes that are not leaves anymore are set to None
                if self.leaf_nodes[ind] == None: continue

                #group is the indicies (boolean mask) that this leaf node has access to
                for dim in range(45):
                    step_size = 1000
                    for i in range(1, step_size):
                        i = i/step_size #test_threshold
                        #determine accuracy of after splitting on ind leaf
                        loss = self._get_accuracy_after_split(features, labels, i, dim, ind)

                        if loss<best["loss"]:
                            best["loss"] = loss
                            best["thresh"] = i
                            best["dim"] = dim
                            best["leaf_index"] = ind
            # create new split, delete leaf node that was split. 
            best_node = self.leaf_nodes[best["index"]]

            #convert the best node to a branch and delete it as a leaf.
            best_node.convertToBranch(thresh = best["thresh"], dim = best["dim"],\
                                      left_ind = len(self.node_list), right_ind = len(self.node_list)+1)

            left, right = self._actual_split(features, labels)

            #add new leaves to nodes and leaves
            self.leaf_nodes[best["index"]] = None
            assert(self.node_list[best["index"]] != None)
            self.node_list += [left, right]
            self.leaf_nodes += [left, right]
            
    def test(self, features_test, labels_test):
        pass


if __name__ == '__main__':
    values = np.array([[1, 3],[2, 5],[3, 6],[4, 7]])
    mask = np.array([True, True, True, False])
    print(values[mask])
    simple = Node(threshold=0.5, dimension=2)
    copy = copy.deepcopy(simple)
    copy.convertToBranch(0.1, 1)
    print(f"copy: {copy}")
    print(f"simple: {simple}")

