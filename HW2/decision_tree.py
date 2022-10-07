import numpy as np
from scipy import stats
from node import Node    
import copy

class DecisionTree:

    def __init__(self):
        self.node_list = []
        
        #keeps track of non-null leaf nodes
        self.leaf_count = 0
        
        # global prediction variable to store predictions for inputs
        self.predictions = np.array([])

    def show_labels(self):
        return [i.getLabel() for i in self.node_list if i.isLeaf()]

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
        if (decision.isLeaf() == True): 
            
            #change the labels of the groups global predictions variable to label of this node
            self.prediction = np.where(group == True, decision.getLabel(), self.prediction)
       
        # not a leaf node, so we must split 
        else: 
            assert(decision.next1!=-1 and decision.next2!=-1)
            decision = self.node_list[node_index]

            # left and right group are the "groups" that can be futher split by the next node
            left_group, right_group = decision.split(features = X, g = group)
            (left_index, right_index) = decision.getNext()

            # TODO: problem!!! keeps old prediction matrix size no bueno
            # need to change 
            self.branch(X, left_index, left_group)
            self.branch(X, right_index, right_group)

    def predict(self, X):
        """
        Returns: predictions for each feature vector based on regions
        """
        #initialize prediction vector
        self.prediction = np.ones(X.shape[1])
        
        group = np.ma.make_mask(self.prediction.shape[0])
        #start at the head node and call recursive branch function
        self.branch(X, 0, group)
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
        leaf = self.node_list[ind]
        left_mask, right_mask = leaf.split(X, threshold, dimension)
        assert(np.array_equal(leaf.getGroup(), left_mask+right_mask))
        left_label = stats.mode(Y[left_mask], keepdims=True)[0]
        right_label = stats.mode(Y[right_mask], keepdims=True)[0]
        loss_for_split = np.sum(np.where(Y[left_mask]!=left_label, 1, 0)) +\
                         np.sum(np.where(Y[right_mask]!=right_label, 1, 0))
        return loss_for_split

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
        def calcLabel(labels, group):
            """
            Inputs: Labels: 1 x N array of labels
                    group: 1 x N boolean array mask 
                        of the only indices in group
            Returns the most occuring label in within the group
            """
            filtered_labels = labels[group]
            return stats.mode(filtered_labels, keepdims=False)[0]

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
    def train(self, features, labels, depth, plot_data = None):
        """
        Creates nodes which represent splits. 
        the thresholds are being trained: t
        the best split is the one that produces the best accuracy 
        across all dimesnsions. 
        The 0-1 loss function for each split is considered. 
        """
        # add the head node
        head = Node(label = stats.mode(labels, keepdims=False)[0], \
                    group = np.ma.make_mask(np.ones(labels.shape[0])),
                    loss = labels.shape[0])
        self.node_list.append(head) #initialize node list to [head]
        self.leaf_count = 1
        while (self.leaf_count < depth):
            print(f"Training iteration for the {self.leaf_count+1} iteration\r\n\
                    ============================================")
            #keep track of the best split to create node later
            best = {
                "loss":features.shape[1],
                "thresh":0, 
                "dim":0,
                "index":0
            }
            #get groups/regions from current tree structure 
            #find best dimension globally
            leaf_tracker = 0
            for ind in range(len(self.node_list)):
                # sinc leaf nodes' indicies are tracked, deleting them from the list is not efficient.
                # the leaf nodes that are not leaves anymore are set to None
                if not self.node_list[ind].isLeaf(): continue
                print(f"Leaf #{leaf_tracker} for depth {self.leaf_count+1}")
                leaf_tracker+=1

                #the loss excluding the leaf of the one we are splitting
                current_loss = sum([i.getLoss() for i in self.node_list if i.isLeaf()]) - self.node_list[ind].getLoss()
                
                #group is the indicies (boolean mask) that this leaf node has access to
                for dim in range(45):
                    # if dim%9 == 0:
                    #     print(f"Training dimension: {dim}/45")
                    step_size = 5
                    for i in range(1, step_size):
                        i = i/step_size #test_threshold
                        #determine accuracy of after splitting on ind leaf
                        loss = current_loss + self._get_accuracy_after_split(features, labels, i, dim, ind)

                        if loss<best["loss"]:
                            best["loss"] = loss
                            best["thresh"] = i
                            best["dim"] = dim
                            best["index"] = ind

            print(f'Loss after best split (dim: {best["dim"]}, thresh:{best["thresh"]}): {best["loss"]}/{labels.shape[0]}')

            # create new split, delete leaf node that was split. 
            best_node = self.node_list[best["index"]]

            #convert the best node to a branch and delete it as a leaf.
            best_node.convertToBranch(thresh = best["thresh"], dim = best["dim"],\
                                      left_ind = len(self.node_list), right_ind = len(self.node_list)+1)

            left, right = self._actual_split(best_node, features, labels)

            #add new leaves to nodes and leaves
            self.node_list += [left, right]
            self.leaf_count += 1

            #sanity check: make sure that nodes are exclusively branches or leaves
            for i in self.node_list:
                if i.isLeaf(): assert(i.isBranch() == False)

            if plot_data == None: continue
            if self.leaf_count%(depth/10) == 0:
                plot_data["train"].append(best["loss"]/labels.shape[0])
                test_loss = self.test(plot_data["features"], plot_data["labels"])
                plot_data["test"].append(test_loss/plot_data["labels"].shape[0])
                        
    def test(self, features_test, labels_test):
        """
        Inputs: features_test - 45 x N
                labels_test - N x 1
        Calculates and prints the total loss 
        """
        prediction = self.predict(features_test)

        loss = np.sum(np.where(prediction!=labels_test, 1, 0))
        print(f"Total loss for test set: {loss}/{labels_test.shape[0]}")
        return loss

if __name__ == '__main__':
    values = np.array([[1, 3],[2, 5],[3, 6],[4, 7]])
    mask = np.array([True, True, True, False])
    print(values[mask])
    simple = Node(threshold=0.5, dimension=2)
    copy = copy.deepcopy(simple)
    copy.convertToBranch(0.1, 1)
    print(f"copy: {copy}")
    print(f"simple: {simple}")

