# Importing Data
X <- read.csv(file="hw03_images.csv", header=FALSE, sep=",")
y <- read.csv(file="hw03_labels.csv", header=FALSE, sep=",")

W <- read.csv(file="initial_W.csv", header=FALSE, sep=",")
v <- read.csv(file="initial_V.csv", header=FALSE, sep=",")

#Get the number of samples
N <- length(y$V1)

# Get the number of classes
K <- max(y)

# Split data into training and testing subsets
X_train = as.matrix(X[1:500,])
X_test = as.matrix(X[-(1:500),])
y_train = as.matrix(y[1:500,])
y_test = as.matrix(y[-(1:500),])

# Apply one hot encoding to the labels
y_train_encoded <- matrix(0, length(y_train), K)
y_train_encoded[cbind(1:length(y_train), y_train)] <- 1

# Define sigmoid function
sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

# Define the safe log function to avoid infinities or unknowns
safelog <- function(x) {
  return (log(x + 1e-100))
}

# Define softmax function
softmax <- function(scores) {
  scores <- exp(scores - matrix(apply(scores, MARGIN = 1, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}

# Define loss function
error <- function(y_truth, y_pred) {
  return (-sum(y_truth * safelog(y_pred)))
}

# Define parameters
eta <- 0.0005
epsilon <- 1e-3
max_iteration <- 500

# Parameter Initialization
Z <- sigmoid(cbind(1, X_train) %*% as.matrix(W))
y_pred <- softmax(cbind(1, Z) %*% as.matrix(v))
objective_values <- -sum(y_train_encoded * safelog(y_pred))

# Iterative Algorithm

# Change the data types
v = as.matrix(v)
W = as.matrix(W)

# Initialize the counter
iteration <- 1
H = 20
while (1) {
  
  for(i in 1:H+1) {
    one_binded_hidden_nodes <- cbind(1,Z)
    v[i,] <- v[i,] + eta * colSums(((y_train_encoded - y_pred) * matrix(one_binded_hidden_nodes[,i], nrow=nrow(y_train_encoded), ncol=ncol(v), byrow=FALSE)))
  }
  for (j in 1:H) {
    W[,j] <- W[,j] + eta * as.matrix(colSums(matrix(rowSums((y_train_encoded - y_pred) * matrix(v[j+1,], nrow=nrow(y_train_encoded), ncol=ncol(v), byrow=TRUE)) * as.matrix(Z[,j]) * as.matrix((1 - Z[,j])), nrow=nrow(y_train_encoded), ncol=nrow(W), byrow=FALSE) * cbind(1,X_train)))
  }
  
  Z <- sigmoid(cbind(1, X_train) %*% W)
  y_pred <- softmax(cbind(1, Z) %*% v)
  objective_values <- c(objective_values, error(y_train_encoded,y_pred))
  
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
}

plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# Get the confusion matrix for train values
y_pred_train <- apply(y_pred, MARGIN = 1, FUN = which.max)
confusion_matrix_train <- table(y_pred_train, y_train)
print(confusion_matrix_train)

# Get the confusion matrix for test values
y_pred_test <- apply(softmax(cbind(1, sigmoid(cbind(1, as.matrix(X_test))%*%W)) %*% v), 1, which.max)
confusion_matrix_test <- table(y_pred_test, y_test)
print(confusion_matrix_test)
