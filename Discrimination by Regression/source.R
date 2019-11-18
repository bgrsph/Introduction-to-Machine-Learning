# Importing Data
X <- read.csv(file="hw02_images.csv", header=FALSE, sep=",")
y <- read.csv(file="hw02_labels.csv", header=FALSE, sep=",")

W <- read.csv(file="initial_W.csv", header=FALSE, sep=",")
w0 <- read.csv(file="initial_w0.csv", header=FALSE, sep=",")

#Get the number of samples
N <- length(y$V1)

# Get the number of classes
K <- max(y)

# Split data into training and testing subsets
X_train = as.matrix(X[1:500,])
X_test = as.matrix(X[-(1:500),])
y_train = as.matrix(y[1:500,])
y_test = as.matrix(y[-(1:500),])

# Make w0 a vector instead of matrix so that R will make addition in sigmoid function
w0 = as.vector(t(w0))

# Apply one hot encoding to the labels
y_train_encoded <- matrix(0, length(y_train), K)
y_train_encoded[cbind(1:length(y_train), y_train)] <- 1

# Define sigmoid function
sigmoid <- function(X, W, w0) {
  return (1 / (1 + exp(-( (X %*% W) + w0))))
}

gradient_W <- function(X, y_truth, y_pred) {
  return (-sapply(X = 1:ncol(y_truth), function(i) colSums(matrix((y_truth[,i] - y_pred[,i])*y_pred[,i]*(1-y_pred[,i]), nrow = nrow(X), ncol = ncol(X), byrow = FALSE) * X)))
}

gradient_w0 <- function(y_truth, y_pred) {
  return (-colSums((y_truth - y_pred) * y_pred * (1-y_pred)))
}

# Set algorithm parameters
eta <- 0.0001
epsilon <- 1e-3
max_iteration <- 500

# Start the iteration
iteration <- 1
objective_values <- c()

test = (X_train %*% as.matrix(W)) + w0
while (1) {
  y_pred <- sigmoid(X_train, as.matrix(W), w0)
  
  objective_values <- c(objective_values, sum((y_train_encoded - y_pred)^2)/2)
  
  W_old <- W
  w0_old <- w0
  
  W <- W - eta * gradient_W(X_train, y_train_encoded, y_pred)
  w0 <- w0 - eta * gradient_w0(y_train_encoded, y_pred)
  
  if (sum((y_train_encoded - y_pred)^2)/2 < epsilon || iteration == max_iteration) {
    break
  }
  
  iteration = iteration + 1
}

# Draw the pilot with respect to errors and iterations
plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Objective Values")

# Get the confusion matrix for train values
y_predicted_train <- apply(y_pred, MARGIN = 1, FUN = which.max)
confusion_matrix_train <- table(y_predicted_train, y_train)
print(confusion_matrix_train)

# Get the confusion matrix for test values
y_predicted_test <- sigmoid(X_test, as.matrix(W), w0)
y_predicted_test <- apply(y_predicted_test, MARGIN = 1, FUN = which.max)
confusion_matrix_test <- table(y_predicted_test, y_test)
print(confusion_matrix_test)

