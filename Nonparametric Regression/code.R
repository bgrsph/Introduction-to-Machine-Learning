# Import the data
data <- read.csv("hw04_data_set.csv")

# Get input and label columns
X <- data$eruptions
y <- data$waiting

# Split the data into training and testing samples
X_train = X[1:150]
y_train = y[1:150]
X_test <- X[151:272]
y_test <- y[151:272]

# Set up the parameters
bin_width = 0.37
origin = 1.5
X_max = max(X)
X_min = min(X)
data_interval <- seq(from = origin, to = X_max, by = 0.001)
borders_left <- seq(from = origin, to = X_max, by = bin_width)
borders_right <- seq(from = origin + bin_width, to = X_max+ bin_width, by = bin_width)

# Define the regressogram function
regressogram <- sapply(1:length(borders_left), function(index) {
  return(mean(y_train[borders_left[index] < X_train & X_train <= borders_right[index]]))
})

# Define a getter for bin number
get_bin_number <- function(index, X_test, origin, bin_width) {
  return ((X_test[index] - origin) / bin_width)
}

# Draw the scatter plot of the train and test data
plot(x=X_train,y=y_train,xlab="Eruption time (min)", ylab="Waiting time to next eruption (min)",
     type = "p", main = paste("h:",bin_width),col = "magenta",las = 1,pch =20
)
points(x=X_test,y=y_test, col = "red",las = 1,pch = 20)

# Apply regressogram function to data
for (bin in 1:length(borders_left)) {
  lines(c(borders_left[bin], borders_right[bin]), c(regressogram[bin], regressogram[bin]), lwd = 1, col = "black")
  if (bin < length(borders_left)) {
    lines(c(borders_right[bin], borders_right[bin]), c(regressogram[bin], regressogram[bin + 1]), lwd = 1, col = "black") 
  }
}

# Get predicted data
i = 1
preds = c()
while (i <= length(X_test)) {
  preds<-append(preds,(y_test[i]-regressogram[get_bin_number(i,X_test, origin, bin_width)])^2)
  i = i+1
}

# Calculate and print the root mean square (RMSE) for test regressogram values
RMSE<-sqrt(sum(preds)/length(X_test))
print(paste("Regressogram => RMSE is", RMSE, "when h is", bin_width))

#########################################Running Mean Smoother########################################

# Define the weight function in the book
b = function(x) {
  if(abs(x) < 0.5)
    return (1)
  else 
    return(0)
}
# Define the running mean smoother function
running_mean_smoother <- sapply(data_interval, function(x) {
  divider = 0
  dividing = 0
  for (i in 1:length(X_train)) {
      weight <- b((x - X_train[i]) / bin_width)
      divider = divider + weight*y_train[i]
      dividing = dividing + weight
  }
  return (divider / dividing)
})

# Draw the scatter plot of the train and test data
plot(x=X_train,y=y_train,xlab="Eruption time (min)", ylab="Waiting time to next eruption (min)",
     type = "p", main = paste("h:",bin_width),col = "magenta",las = 1,pch = 20)

points(x=X_test,y=y_test, col = "red",las = 1,pch = 20)

# Apply RMSE regressogram function to data
for (bin in 1:length(data_interval)) {
  lines(c(data_interval[bin], data_interval[bin+1]), c(running_mean_smoother[bin], running_mean_smoother[bin]), lwd = 1, col = "black")
  if (bin < length(data_interval)) {
    lines(c(data_interval[bin], data_interval[bin+1]), c(running_mean_smoother[bin], running_mean_smoother[bin + 1]), lwd = 1, col = "black") 
  }
}

# Get predicted data
preds = c()
for (i in 1: length(X_test)) {
  preds<-append(preds,(y_test[i]-running_mean_smoother[(X_test[i]-origin) / 0.001])^2)
}
# Calculate and print the root mean square (RMSE) for test RMSE values
RMSE_RMS = sqrt(sum(preds)/length(X_test)) 
print(paste("Running Mean Smoother => RMSE is", RMSE_RMS, "when h is", bin_width))

#########################################Kernel Smoother###########################################

# Define the kernel function in the book

kernel =function(x){
  return ((1 / sqrt(2 * pi)) * exp(-(x ** 2) / 2))
  }

# Define the running mean smoother function
kernel_smoother <- sapply(data_interval, function(x) {
  divider = 0
  dividing = 0
  for (i in 1:length(X_train)) {
    weight <- kernel((x - X_train[i]) / bin_width)
    divider = divider + weight*y_train[i]
    dividing = dividing + weight
  }
  return (divider / dividing)
})

plot(x=X_train,y=y_train,xlab="Eruption time (min)", ylab="Waiting time to next eruption (min)",
     type = "p", main = paste("h:",bin_width),col = "magenta",las = 1,pch = 20)
points(x=X_test,y=y_test, col = "red",las = 1,pch = 20)

for (bin in 1:length(data_interval)) {
  lines(c(data_interval[bin], data_interval[bin+1]), c(kernel_smoother[bin], kernel_smoother[bin]), lwd = 1, col = "black")
  if (bin < length(data_interval)) {
    lines(c(data_interval[bin], data_interval[bin+1]), c(kernel_smoother[bin], kernel_smoother[bin]), lwd = 1, col = "black") 
  }
}

# Get predicted data
preds = c()
for (i in 1: length(X_test)) {
  preds<-append(preds,(y_test[i]-kernel_smoother[(X_test[i]-origin) / 0.001])^2)
}
RMSEforKS<-sqrt(sum(preds)/length(X_test))
print(paste("Kernel Smoother => RMSE is", RMSEforKS, "when h is", bin_width))

