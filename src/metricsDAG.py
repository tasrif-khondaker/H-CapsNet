# Hierarchical metrics for DAGs (directed acyclic graphs)

import numpy as np


# Compute hierarchical precision
def hierarchical_precision(y_true, y_pred):
    num_examples = len(y_true[0])
    hP = 0
    for j in range(num_examples):
        y_true_aug = set()
        y_pred_aug = set()
    
        for i in range(len(y_true)): 
            y_true_argmax = np.argmax(y_true[i][j])
            y_true_aug.add(y_true_argmax)
            
            y_pred_argmax = np.argmax(y_pred[i][j])
            y_pred_aug.add(y_pred_argmax)

        hP += len(y_true_aug & y_pred_aug) / len(y_pred_aug)
    
    hP /= num_examples
    return hP

# Compute hierarchical recall 
def hierarchical_recall(y_true, y_pred):

  num_examples = len(y_true[0])
  hR = 0
  
  for j in range(num_examples):

    y_true_aug = set()
    y_pred_aug = set()

    for i in range(len(y_true)):
      
      y_true_argmax = np.argmax(y_true[i][j])  
      y_true_aug.add(y_true_argmax)
               
      y_pred_argmax = np.argmax(y_pred[i][j])
      y_pred_aug.add(y_pred_argmax)

    hR += len(y_true_aug & y_pred_aug) / len(y_true_aug)
  
  hR /= num_examples

  return hR

def hierarchical_f1(y_true, y_pred):
  
  p = hierarchical_precision(y_true, y_pred)
  r = hierarchical_recall(y_true, y_pred)
  
  if p + r == 0:
    f1 = 0
  else:
    f1 = 2 * (p * r) / (p + r)
  
  return f1

# Compute hierarchical consistency without tree
# def hierarchical_consistency(y_true, y_pred):

#   num_examples = len(y_true[0])
#   consistency = 0

#   for j in range(num_examples):

#     consistent = 1
    
#     for i in range(len(y_true) - 1):
    
#       true_parent = np.argmax(y_true[i][j])
#       pred_parent = np.argmax(y_pred[i][j])  
      
#       true_child = np.argmax(y_true[i+1][j])
#       pred_child = np.argmax(y_pred[i+1][j])
      
#       if true_parent != pred_parent:
#         consistent = 0
#         break
        
#       if pred_parent != pred_child and true_parent != true_child:
#         consistent = 0
#         break
        
#     consistency += consistent
  
#   return consistency / num_examples
def hierarchical_consistency(y_true, y_pred):

  # Get max index from all levels
  max_index = max([np.argmax(y).max() for y in y_true])
  
  # Create adjacency matrix
  adj_matrix = np.zeros((max_index+1, max_index+1))
  
  for j in range(len(y_true[0])):
    for i in range(len(y_true)-1):
      child = np.argmax(y_true[i+1][j])  
      parent = np.argmax(y_true[i][j])
      
      # Check bounds before assigning
      if 0 <= parent < adj_matrix.shape[0] and 0 <= child < adj_matrix.shape[1]:
        adj_matrix[parent, child] = 1

  num_examples = len(y_true[0])
  consistency = 0
  
  for j in range(num_examples):

    consistent = 1

    for i in range(len(y_true) - 1):

      pred_parent = np.argmax(y_pred[i][j])
      pred_child = np.argmax(y_pred[i+1][j])
      
      if 0 <= pred_parent < adj_matrix.shape[0] and 0 <= pred_child < adj_matrix.shape[1]:
        if adj_matrix[pred_parent, pred_child] == 0:
          consistent = 0
          break

    consistency += consistent
  
  return consistency / num_examples
  

def hierarchical_exact_match(y_true, y_pred):

  num_examples = len(y_true[0])
  exact_match = 0
  
  for j in range(num_examples):
    
    match = 1
    
    for i in range(len(y_true)):
    
      if np.argmax(y_true[i][j]) != np.argmax(y_pred[i][j]):
        match = 0
        break
      
    exact_match += match  
    
  return exact_match / num_examples

# Calculate hierarchical measurements
def hierarchical_metrics(y_true, y_pred):
    return {
        "hierarchical precision": hierarchical_precision(y_true, y_pred),
        "hierarchical recall": hierarchical_recall(y_true, y_pred),
        "hierarchical f1": hierarchical_f1(y_true, y_pred),
        "hierarchical consistency": hierarchical_consistency(y_true, y_pred),
        "hierarchical exact match": hierarchical_exact_match(y_true, y_pred)
    }