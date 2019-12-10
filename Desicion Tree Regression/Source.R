# Import the data
data <- read.csv("hw05_data_set.csv")

# Get input and label columns
X <- data$eruptions
y <- data$waiting

# Split the data into training and testing samples
X_train = X[1:150]
y_train = y[1:150]
X_test <- X[151:272]
y_test <- y[151:272]

# Number of samples
N = length(data)
N_train = length(X_train)
N_test = length(X_test)

# Number of classes
K = max(y)

infer_tree = function(P, dataset) {
  # Create necessary data structures
  node_indices <- list()
  is_terminal <- c()
  need_split <- c()
  node_features <- c()
  node_splits <- c()
  node_frequencies <- list()
  node_prediction <- list()
  y_predictions <- c()
  
  # Put all training instances in root node
  node_indices <- list(1:N_train)
  is_terminal <- c(FALSE)
  need_split <- c(TRUE)
  
  # Training
  while(TRUE) {
    # find nodes that need splitting
    split_nodes <- which(need_split)
    
    # check whether we reach all terminal nodes
    if (length(split_nodes) == 0) {
      break
    }
    
    # find best split positions for all nodes
    for (split_node in split_nodes) {
      data_indices = node_indices[[split_node]]
      need_split[split_node] = FALSE
      node_prediction[[split_node]]  <- sum(y_train[data_indices])/length(data_indices)   
      
      # check whether node is pure with pre-prun parameter
      if (length(unique(y_train[data_indices])) == 1 || length(data_indices) <= P) {
        is_terminal[split_node] <- TRUE
      } else {
        is_terminal[split_node] <- FALSE
        best_score = 0
        best_split = 0
        unique_values = sort(unique(X_train[data_indices]))
        split_positions = (unique_values[-1] + unique_values[-length(unique_values)]) / 2
        split_scores = rep(0, length(split_positions))
        
        for (index in 1:length(split_positions)) {
          left_indices <- data_indices[which(X_train[data_indices] < split_positions[index])]
          right_indices <- data_indices[which(X_train[data_indices] >= split_positions[index])]
          split_scores[index] <- sum(sapply(list(left_indices, right_indices), function(indices){ sum((y_train[indices] - sum(y_train[indices])/length(indices))^2) })) / length(data_indices)
        }
        # decide where to split on which feature
        best_score = min(split_scores)
        best_split = split_positions[which.min(split_scores)]
        node_splits[split_node] <- best_split
        
        # create left node using the selected split
        left_indices <- data_indices[which(X_train[data_indices] < best_split)]
        node_indices[[2 * split_node]] <- left_indices
        is_terminal[2 * split_node] <- FALSE
        need_split[2 * split_node] <- TRUE
        
        # create left node using the selected split
        right_indices <- data_indices[which(X_train[data_indices] >= best_split)]
        node_indices[[2 * split_node + 1]] <- right_indices
        is_terminal[2 * split_node + 1] <- FALSE
        need_split[2 * split_node + 1] <- TRUE
      }
    }
  }
  for (j in dataset) {
    index = 1
    while(TRUE) {
      if (is_terminal[index] == TRUE) {
        y_predictions = c(y_predictions,node_prediction[[index]])
        break
      } else {
        if (j <= node_splits[index]) {
          index <- index * 2
        } else {
          index <- index * 2 + 1
        }
      }
    }
  }
  
  return(y_predictions)
}

# Train a desicion tree for P = 25
P = 25
y_pred = rep(0, N_test)
y_pred = infer_tree(P, X_test)

# Print the Root Mean Square Error with respect to P = 25
print(paste("RMSE is", sqrt(mean((y_test - y_pred)^2)), "when P is", P))


# NOTE: I couldn't manage to show my fit in the drawings, but RMSE is exactly the same with assignment definition.

data_interval = seq(from=  5, to = 50, by = 5)
p_testing <- sapply(data_interval, function(P){
  y_predicted <- infer_tree(P, X_test)
  return (sqrt (mean((y_test - y_predicted)^2)))
})

plot(1:length(p_testing), p_testing, type="b", col = "black", ylab = "RMSE", xlab = "Pre−pruning size (P)")


