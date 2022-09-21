from xml.sax.handler import feature_external_ges
import numpy as np
from scipy import stats
from node import Node    

class DecisionTree:

    def __init__(self, depth):
        self.depth = depth
        self.node_list = []
        self.leaf_nodes = []
        
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
            left_group, right_group = decision.split(X, group)
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

    def get_accuracy(self, X: np.array, Y: np.array, threshold: float, dimension: int)->float:
        """
        Returns: the 0-1 loss of the dimensional threshold. 
            Must iterate through the predictions and compare
            actual label (Y) and the predicted label(majority label per region)
        Inputs: features: 45 x N
                labels: N x 1
        Output: total correct/total_incorrect
        Procedure: all the leaves are stored in leaf list, so only the target
                   node is split and its accuracy is calculated. 
        """
        
        #pass through decsion tree nodes. update predictions
        pred_Y = self.predict(X)
        #compare to actual
        loss = np.sum(np.where(Y!=pred_Y, 1, 0))
        return loss

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
        while (self.leaf_nodes.size() < depth):
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
                #group is the indicies (boolean mask) that this leaf node has access to
                group = self.leaf_nodes[ind].getGroup()
                for dim in range(45):
                    step_size = 1000
                    for i in range(1, step_size):
                        i = i/step_size #test_threshold
                        #determine accuracy of threshold.    
                        loss = self.get_accuracy(features, labels, i, dim, group)
        #               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # TODO: this is wrong... how do we actually find the best accuracy? 
        # TODO: the problem: No where is label set for the leaf nodes
        #   - need the group to filter out only labels for this group. 
                        if loss<best["loss"]:
                            best["loss"] = loss
                            best["thresh"] = i
                            best["dim"] = dim
                            best["leaf_index"] = ind
            # create new split, delete leaf node that was split. 
            left = Node(best["thresh"])
            del self.leaf_nodes[best["leaf_index"]]
            


    def test(self, features_test, labels_test):
        pass

if __name__ == '__main__':
    DecisionTree()
