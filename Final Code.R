######################## EDA ########################
##################### Initializing Environment #####################
setwd("/Users/dongyangwang/Desktop/UW/First Year Grad/Spring 2022/Stat 527")
rm(list=ls())

# Library
# EDA
library(tidyverse)
library(corrplot)
library(GGally)

# Create Dummy Variables
library(fastDummies)

# Neural network
# install.packages("devtools")
# library(devtools)
# devtools::install_github("bips-hb/neuralnet")
library(neuralnet)

# KNN
library(class)

# SVM
library(e1071)

# Naive Bayes
library(caret)
library(bayesloglin)
library(caTools)

# Random Forest
library(randomForest)
library(fields)
library(ranger)
library(InformationValue)

# Decision Trees
library(rpart)
library(rpart.plot)

# Evaluation
library(yardstick)
library(pROC)
library(ROCR)

# Setting Seed for Reproducibility
set.seed(42)

# Data Loading
raw_data<-read.csv("adult.csv")

##################### Structure of Dataset #####################

# Structure of the Dataset
str(raw_data)
summary(raw_data)

##################### Removing NA #####################

income_data <- raw_data
income_data[income_data == "?"] <- NA
income_data <- drop_na(income_data)

##################### Feature Engineering #####################

# Response Variable: Two Levels
income_data$income[income_data$income == ">50K"] <- 1
income_data$income[income_data$income == "<=50K"] <- 0
income_data$income <- as.numeric(income_data$income)

# Other Variable Transformation
income_data$age<-as.numeric(income_data$age)

# Conversion to Categorical Data
income_data$workclass<-as.factor(income_data$workclass)
income_data$marital.status<-as.factor(income_data$marital.status)
income_data$occupation<-as.factor(income_data$occupation)
income_data$relationship<-as.factor(income_data$relationship)
income_data$race<-as.factor(income_data$race)
income_data$gender<-as.factor(income_data$gender)

# Grouping Country Origins

income_data$native.country<-as.character(income_data$native.country)

north.america <- c("Canada", "Cuba", "Dominican-Republic", "El-Salvador", "Guatemala",
                   "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua",
                   "Outlying-US(Guam-USVI-etc)", "Puerto-Rico", "Trinadad&Tobago",
                   "United-States")
asia <- c("Cambodia", "China", "Hong", "India", "Iran", "Japan", "Laos",
          "Philippines", "Taiwan", "Thailand", "Vietnam")
south.america <- c("Columbia", "Ecuador", "Peru")
europe <- c("England", "France", "Germany", "Greece", "Holand-Netherlands",
            "Hungary", "Ireland", "Italy", "Poland", "Portugal", "Scotland",
            "Yugoslavia")
other <- c("South")

income_data$native.country[income_data$native.country %in% north.america] <- "North America"
income_data$native.country[income_data$native.country %in% asia] <- "Asia"
income_data$native.country[income_data$native.country %in% south.america] <- "South America"
income_data$native.country[income_data$native.country %in% europe] <- "Europe"
income_data$native.country[income_data$native.country %in% other] <- "Other"

table(income_data$native.country)

# Modify column names for easier data manipulation
names(income_data) <- str_replace_all(names(income_data),"-", "_")
names(income_data) <- str_replace_all(names(income_data),"\\.", "_")
names(income_data) <- str_replace_all(names(income_data)," ", "_")
names(income_data) <- str_replace_all(names(income_data),"native.country", "country")

##################### Elementary Visualizations #####################

# Histograms for Main Variables
hist(income_data$age)
hist(income_data$fnlwgt)
hist(income_data$educational_num)
hist(income_data$capital_gain)
hist(income_data$capital_loss)
hist(income_data$hours_per_week)

# Summary
summary(income_data)

##################### Correlation & Interaction #####################

# Pearson's Correlation Coefficients
Pearson=cor(select(income_data, 
                   c(age, fnlwgt, educational_num, capital_gain, capital_loss,
                     hours_per_week, income)), method = c("pearson"))

symnum(Pearson, abbr.colnames = FALSE)

corrplot(Pearson, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

# Interaction Visualizations Omited, since classification problem won't automate

######################## Model Training ########################

# Train & Test Split
total_row = nrow(income_data)
split <- sample(total_row, total_row*0.8)

train <- income_data[split,]
test<- income_data[-split,]

# Split train dataset into training set and validation set
train_row = nrow(train)
split_val <- sample(train_row, train_row*0.8)

model_train <- train[split_val,]
validation<- train[-split_val,]

##################### Logistic Regression ##################### 
# Build the regression model based on training set
# Specify a null model with no predictors
null_model <- glm(income ~ 1, data = model_train, family = "binomial")

# Specify the full model using all of the potential predictors
full_model <- glm(income ~ .-education, data = model_train, family = "binomial")

# Use a forward stepwise algorithm to build a parsimonious model
step_model <- step(null_model, scope = list(lower = null_model, upper = full_model), direction = "both")

# The summary below shows that all variables should be included based on AIC
summary(step_model)

# Utilize validation set to determine threshold with best accuracy

# Function to return accuracy for different threshold
valid_threshold <- function(data, model, th = 0.5){
  
  # Create vector to store results
  k<-length(th)
  acc <- rep(NA, k)
  
  predict_reg <- predict(model, newdata = data, type = "response")
  actual_response<-data$income
  
  for (j in 1:k) {
    predict_response <- ifelse(predict_reg >th[j], 1, 0)
    
    # Build the confusion matrix
    outcomes_logistic <- table(actual_response, predict_response)
    confusion <- conf_mat(outcomes_logistic)
    
    # Get accuracy of the model on validation set
    acc[j]<-summary(confusion, event_level = "second")[1,3]$.estimate
  }
  
  threshold<-th[which.max(acc)]
  
  return(list(threshold<-threshold, accuracy_on_valid<-acc))
}

th<-c(0.3,0.4,0.5,0.6,0.7)
th_and_accuracy<-valid_threshold(data=validation, model=step_model, th=th)

# For now, we have determined the regression model and the threshold, then we 
# Test the performance of this logistic classification model on test set

obtain_metric<-function(model,threshold, test_data){
  pred <- predict(model, newdata = test_data, type = "response")
  
  predicted_response <- ifelse(pred> threshold, 1, 0)
  
  actual_response<-test_data$income
  
  # "Automatically" plot the confusion matrix
  outcomes_ <- table(actual_response, predicted_response)
  confusion <- conf_mat(outcomes_)
  metric<-summary(confusion, event_level = "second")$.estimate
  accuracy<-metric[1]
  sensitivity_value<-metric[3]
  specificity_value<-metric[4]
  
  # Plot the ROC curve and calculate auc
  ROC_logistic <- roc(actual_response, pred)
  plot(ROC_logistic, col = "red")
  auc_value<-as.numeric(auc(ROC_logistic))
  newlist<-list(accuracy, sensitivity_value, specificity_value, auc_value)
  
  cat("this function returns accuracy, sensitivity, specificity and auc in order \n")
  
  return(newlist)
}


logistic_metric<-obtain_metric(step_model,threshold = th_and_accuracy[[1]], test)
#logistic_metric

##################### Naive Bayes##################### 
# Build multinomial distribution 
discrete_distribution <- function(fieldName,training_set,label){
  all_fields <- colnames(training_set)
  x <- training_set[which(training_set[,length(all_fields)] == label),]
  x <- x[,which(all_fields == fieldName)]
  
  unique_elements <- levels(x)
  distribution <- t(data.frame(count = rep(NA,length(unique_elements))) )
  colnames(distribution) <- unique_elements
  
  for(i in 1:length(unique_elements)){
    distribution[i] <- length(x[x==unique_elements[i]])
  }
  
  distribution <- distribution/sum(distribution)
  
  return(distribution)
}

# Build KDE on continuous predictor 
continuous_distribution <- function(fieldName,training_set,label){
  all_fields <- colnames(training_set)
  x <- training_set[which(training_set[,length(all_fields)] == label),]
  
  x <- x[,which(all_fields == fieldName)]
  kde <- density(x,bw = "ucv",kernel="gaussian",n = 800)
  interpolate_x <- kde$x
  interpolate_y <- kde$y
  
  density_estimation <- splinefun(interpolate_x,interpolate_y,method="natural")
  
  return(density_estimation)
}

# Predict joint probability of all continuous/discrete random variables 
get_prediction <- function(a_grid,distribution,training_set,data,
                           label_distribution,fieldNames){
  
  posterior_prob_c1 <- log(label_distribution$C1)
  posterior_prob_c0 <- log(label_distribution$C0)
  prob <- rep(NA,nrow(a_grid))
  
  # Use all pre-built distribution to calculate probability of 
  # a single data point with nrow(a_grid)/2 features
  for(i in 1:nrow(a_grid)){
    current_field <- a_grid[i,]$fieldName
    index <- which(fieldNames == current_field)
    label <- a_grid[i,]$income
    distri <- distribution[[i]]
    information <- data[1,index]
    
    if(is.numeric(training_set[1,index])==TRUE ){
      
      if(distri(information) > 0){
        prob[i] <- log(distri(information))
      }else{
        prob[i] <- 1/nrow(training_set)
      }
      
    }else if(is.numeric(training_set[1,index])==FALSE ){
  
      categories <- colnames(distri)
      prob[i] <- log(distri[which(categories == information)]) 
      
    }
    
  }
  
  # Split prob into two vector for later use
  prob_c1 <- rep(NA,nrow(a_grid)/2)
  prob_c0 <- rep(NA,nrow(a_grid)/2)
  
  for(i in 1: length(prob)){
    if(i < length(prob)/2 || i == length(prob)/2){
      prob_c1[i] <- prob[i]
    }else{
      prob_c0[i-length(prob)/2] <- prob[i]
    }
  }
  
  # Compute log likelihood of class 1 and class 0
  posterior_prob_c1 <- posterior_prob_c1 + sum(prob_c1) 
  posterior_prob_c0 <- posterior_prob_c0 + sum(prob_c0)
  
  if(posterior_prob_c1 > posterior_prob_c0){
    return(data.frame(class=1,prob= posterior_prob_c1-posterior_prob_c0) )
  }else{
    return(data.frame(class=0,prob=posterior_prob_c0-posterior_prob_c1) )
  }
}

obtain_metric_naivebayes<-function(predicted_class, test_set){
  
  expected_class <- test_set$income
  t <- table(expected_class,predicted_class)
  confusion.matrix <- summary(conf_mat(t))
  
  accuracy <- confusion.matrix[1,3]$.estimate
  
  Sensitivity <- confusion.matrix[3,3]$.estimate
  
  Specificity <- confusion.matrix[4,3]$.estimate
  
  t_ROC <- roc(test_set$income, score,levels = c(0, 1), direction = ">")
  plot(t_ROC, col = "red")
  AUC <- as.numeric(auc(t_ROC))
  newlist<-list(accuracy, Sensitivity, Specificity, AUC)
  
  cat("this function returns accuracy, sensitivity, specificity and auc in order \n")
  return(newlist)
}

# Read data and split into training, and test set
# Specific data pre-processing for Naive Bayes
drop <- c("education")
df1 = income_data[,!(names(income_data) %in% drop)]
df1$educational_num <- as.numeric(df1$educational_num)
df1$country <- as.factor(df1$country)

#split the data in the same way.
total_row = nrow(df1)
split <- sample(total_row, total_row*0.8)
training_set <- df1[split,]
test_set<- df1[-split,]

P_C1 <- sum(training_set$income == 1)
P_C0 <- sum(training_set$income == 0)
label_distribution <- data.frame(C1= P_C1/nrow(training_set), C0=P_C0/nrow(training_set) )

fieldNames <- colnames(training_set)
fieldNames <- fieldNames[1:length(fieldNames)-1]
randomVariable <- rep(NA, length(fieldNames))

binary_outcome <- c(1,0)
a_grid <- expand.grid(fieldName = fieldNames,income = binary_outcome)
distribution <- list()

# For each 
for(i in 1:nrow(a_grid)){
  current_field <- a_grid[i,]$fieldName
  index <- which(fieldNames == current_field)
  label <- a_grid[i,]$income
  if(is.numeric(training_set[1,index])==TRUE ){
    distribution[[length(distribution)+1]] <- 
      continuous_distribution(current_field,training_set,label)
  }else{
    distribution[[length(distribution)+1]] <-
      discrete_distribution(current_field,training_set,label)
  }
}

predicted_class <- rep(NA,nrow(test_set))
score <- rep(NA,nrow(test_set)) 

for(j in 1: nrow(test_set)){
  current_data <- test_set[j,]
  result <- get_prediction(a_grid,distribution,training_set,
                           current_data,label_distribution,fieldNames)
  predicted_class[j] <- result$class
  score[j] <- result$prob
}

naivebayes_metric<-obtain_metric_naivebayes(predicted_class, test_set = test_set)
#naivebayes_metric

##################### KNN ##################### 

# Change to better variable types
knn_data <- income_data

# Dummies
# Need numeric types for knn to work, not factors or characters

# Converting Factor Variables
# Make dummy variables of 6 columns
knn_data <- dummy_cols(knn_data, 
                       select_columns = c('workclass', 'marital_status', "occupation",
                                          "relationship", "race", "gender", "country"))

# Leaving out unnecessary variables
knn_data <- subset(knn_data, select = - c(workclass, marital_status, occupation,
                                          relationship, race, gender, education, country))

knn_data$income <- as.numeric(knn_data$income)

train_knn <- knn_data[split,]
test_knn <- knn_data[-split,]

# Use CV to find the optimal number of neighbors

neighbors <- seq(10, 30, length.out = 11)
numbers <- seq(1:length(neighbors))

# Cross validate (5 folds) to find the best number of neighbors:
# Here the training set is divided into training and validation sets
cv_accurracy_knn <- function(data = train_knn, i, num = 5){
  
  # Create folds randomly
  n <- nrow(data)
  folds <- sample(rep(1:num, length = n))
  
  # Create vector to store results
  accuracy <- rep(NA, num)
  for(j in 1:num){
    
    # Train model
    train_cv <- folds != j
    data_train <- data[train_cv, ]
    data_test <- data[!train_cv, ]
    
    cv_model <- class::knn(data_train[,-7],data_test[,-7], 
                    cl = data_train$income, k = neighbors[i])
    
    # Compute accuracy on fold j (not used for training)
    accuracy[j]  <- mean(cv_model == data_test$income)
  }
  # Compute average mse
  accuracy_cv <- mean(accuracy)
  return(accuracy_cv)
}

# CV for best number of neighbors on the training set

accuracy_knn <- sapply(numbers, function(number) cv_accurracy_knn(data = train_knn, i = number, num = 5))

# Best number of neighbors
best_knn <- neighbors[which.max(accuracy_knn)]

# Fit the model on test data
knn_model <- class::knn(train_knn[,-7],test_knn[,-7], 
                 cl = train_knn$income, k = best_knn, prob =TRUE)
class::knn(train_knn[,-7],test_knn[,-7], cl = train_knn$income, use.all = TRUE)

obtain_metric_knn<-function(knn_model,test_knn){
  
  # First we use the confusion matrix
  actual_knn <- test_knn$income
  
  # "Automatically" plot the confusion matrix
  outcomes_knn <- table(actual_knn, knn_model)
  confusion_knn <- conf_mat(outcomes_knn)
  autoplot(confusion_knn)
  
  # Get summary metrics
  metric<-summary(confusion_knn, event_level = "second")$.estimate
  accuracy<-metric[1]
  sensitivity_value<-metric[3]
  specificity_value<-metric[4]
  
  # Then we use the ROC/AUC
  predicted_knn <- attributes(knn_model)$prob
  
  # Plot the ROC of the KNN model
  ROC_knn <- roc(actual_knn, predicted_knn)
  plot(ROC_knn, col = "red")
  auc_value<-as.numeric(auc(ROC_knn))
  
  newlist<-list(accuracy, sensitivity_value, specificity_value, auc_value)
  cat("this function returns accuracy, sensitivity, specificity and auc in order \n")
  return(newlist)
}

knn_metric<-obtain_metric_knn(knn_model, test_knn)
knn_metric


##################### SVM ##################### 
set.seed(42)
# New dataset for use
svm_data <- train

# Normalize data for the model
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Normalization will significantly improve speed for running the data
svm_data$age <- normalize(svm_data$age)
svm_data$fnlwgt <- normalize(svm_data$fnlwgt)
svm_data$educational_num <- normalize(svm_data$educational_num)
svm_data$capital_gain <- normalize(svm_data$capital_gain)
svm_data$capital_loss <- normalize(svm_data$capital_loss)
svm_data$hours_per_week <- normalize(svm_data$hours_per_week)
svm_data$country <- as.factor(svm_data$country)

# Create Validation Set
valid <- sample(nrow(svm_data), 0.8*nrow(svm_data))
svm_train <- svm_data[valid,]
svm_valid <- svm_data[-valid,]

# Possible costs to test
cost_svm = c(0.01, 0.1, 1, 10, 100)

# Length indicator
index_cost_svm <- seq(1:length(cost_svm))

# Storage of accuracy
accuracy_svm <- rep(NA, length(cost_svm))

# Used Train-Validate approach instead of the tune.svm function because 
# that was too computationally expensive

# Linear
for (i in 1:length(cost_svm)){
  cv_model <- svm(income ~ .-education, data = svm_train, 
                  type = "C-classification", cost = cost_svm[i], kernel = "linear")
  prediction <- predict(cv_model, svm_valid)
  accuracy_svm[i] <- mean(prediction == svm_valid$income)
}
max_linear <- max(accuracy_svm)
linear_svm_cost <- cost_svm[which.max(accuracy_svm)]

# Polynomial
for (i in 1:length(cost_svm)){
  cv_model <- svm(income ~ .-education, data = svm_train, 
                  type = "C-classification", cost = cost_svm[i], kernel = "polynomial")
  prediction <- predict(cv_model, svm_valid)
  accuracy_svm[i] <- mean(prediction == svm_valid$income)
}

max_poly <- max(accuracy_svm)
poly_svm_cost <- cost_svm[which.max(accuracy_svm)]

# Sigmoid
for (i in 1:length(cost_svm)){
  cv_model <- svm(income ~ .-education, data = svm_train, 
                  type = "C-classification", cost = cost_svm[i], kernel = "sigmoid")
  prediction <- predict(cv_model, svm_valid)
  accuracy_svm[i] <- mean(prediction == svm_valid$income)
}

max_sigmoid <- max(accuracy_svm)
sigmoid_svm_cost <- cost_svm[which.max(accuracy_svm)]

# Get the best accuracy
kernel_svm <- c('linear', 'polynomial', 'sigmoid')
max_svm <- c(linear_svm_cost, poly_svm_cost, sigmoid_svm_cost)
cost_all_svm <- c(max_linear, max_poly, max_sigmoid)

# Best kernel and cost based on accuracy
svm_kernel <- kernel_svm[which.max(cost_all_svm)]
svm_cost <- max_svm[which.max(cost_all_svm)]

# So we will determine our final model based on accuracy of the respective models
svm_model <- svm(income ~ .-education, data = train, fitted = T, probability = T,
                 type = "C-classification", cost = svm_cost, kernel = svm_kernel)

obtain_metric_svm<-function(svm_model, test_data){
  
  # Evaluate on test set
  pred_svm <- predict(svm_model, test_data, probability=T)
  pred_svm_prob <- attributes(pred_svm)$prob[,2]
  
  # "Automatically" plot the confusion matrix
  outcomes_svm <- table(test_data$income, pred_svm)
  confusion_svm <- conf_mat(outcomes_svm)
  autoplot(confusion_svm)
  
  # Get summary metrics
  metric<-summary(confusion_svm, event_level = "second")$.estimate
  accuracy<-metric[1]
  sensitivity_value<-metric[3]
  specificity_value<-metric[4]
  
  # Plot the ROC of the SVM model
  ROC_svm <- roc(test_data$income, pred_svm_prob)
  plot(ROC_svm, col = "red")
  auc_value<-as.numeric(auc(ROC_svm))
  newlist<-list(accuracy, sensitivity_value, specificity_value, auc_value)
  
  cat("this function plot confusion matrix and roc curve, returns accuracy, sensitivity, specificity and auc in order \n")
  return(newlist)
}

svm_metric<-obtain_metric_svm(svm_model, test)
#svm_metric

##################### Decision Tree ##################### 
# CV to determine the size of the tree by confirming parameter cp, this time 
# we shall use the whole set "train"
cv_cp <- function(data, cp, k = 5){
  
  # Create folds randomly
  n <- nrow(data)
  folds <- sample(rep(1:k, length = n))
  
  # Create vector to store results
  auc <- rep(NA, k)
  for(j in 1:k){
    
    # Train model on all folds except j
    train <- folds != j
    tree_model <- rpart(income ~ .-education, data = data[train, ], 
                        method="class", minsplit = 200,cp=cp)
    
    # Compute AUC on fold j (not used for training)
    pred <- predict(tree_model, 
                    newdata = data[!train, ], type = "prob")
    ROC_tree <- roc(data$income[!train], pred[,2])
    plot(ROC_tree, col = "red")
    auc[j]<-as.numeric(auc(ROC_tree))
  }
  # Compute average auc
  auc.cv <- mean(auc)
  return(auc.cv)
}

cp<-c(0.001, 0.0007, 0.0005, 0.0003, 0.0001, 0.00007, 0.00005, 0.00003)
valid_result <- sapply(cp, function(cp) cv_cp(train, cp))
cp<-cp[which.max(valid_result)]

# Function to build tree model with the most suitable size, which is determined before
fbtree_mod<-function(data_train, cp){
  tree_model <- rpart(income ~ .-education, data = data_train, 
                      method="class", minsplit = 200,cp=cp)
  #draw structure plot
  rpart.plot(tree_model)
  return(tree_model)
}

tree_model<-fbtree_mod(train,cp)

# Function to obtain metrics to determine tree model's performance on test set.
obtain_metric_tree <- function(model, data_test){
  
  # Test
  pred <- predict(model, 
                  newdata = data_test, type = "class")
  outcomes <- table(data_test$income, pred)
  confusion <- conf_mat(outcomes)
  metric<-summary(confusion, event_level = "second")$.estimate
  accuracy<-metric[1]
  sensitivity_value<-metric[3]
  specificity_value<-metric[4]
  
  # To compute auc and draw roc curve plot
  pred <- predict(model, 
                  newdata = data_test, type = "prob")
  ROC_tree <- roc(data_test$income, pred[,2])
  plot(ROC_tree, col = "red")
  auc_value<-as.numeric(auc(ROC_tree))

  newlist<-list(accuracy, sensitivity_value, specificity_value, auc_value)
  cat("this function returns accuracy, sensitivity, specificity and auc in order \n")
  return(newlist)
}

tree_metric<-obtain_metric_tree(tree_model, test)
#tree_metric

##################### Random Forest ##################### 
# Data preprocessing for random forest
return_split <- function(income_data){

  income_data$country <- as.factor(income_data$country)
  income_data$capital_gain <- normalize(income_data$capital_gain)
  income_data$capital_loss <- normalize(income_data$capital_loss)
  
  drop <- c("education")
  df1 = income_data[,!(names(income_data) %in% drop)]
  
  # Split dataset 
  spec = c(train = .64, test = .2, validate = .16)
  
  g = sample(cut(
    seq(nrow(df1)), 
    nrow(df1)*cumsum(c(0,spec)),
    labels = names(spec)
  ))
  
  res = split(df1, g)
  
  return(list(train_reg=res$train,validation_reg=res$validate,test_reg=res$test))
}

# Split into training, validation, and test set with the proportion of
# 0.64,0.16,0.2. Besides, we normalize capital.gain and capital.loss
set.seed(42)

# return_split function needs the function normalize, which is defined in the SVM section
split <- return_split(income_data)
training_set <- split[[1]]
validation_set <- split[[2]]
test_set <- split[[3]]

# Check correlation between categorical variables by Chi squared test and only store the p value.

categorical_variables <- c("workclass","marital_status","occupation",
                           "relationship","race","gender","country")
max <- length(categorical_variables)
all_p_values <- data.frame(matrix(rep(NA,max^2),nrow=max))
colnames(all_p_values) <- categorical_variables
rownames(all_p_values) <- categorical_variables

return_chi2_test <- function(df,var1,var2){
  all_labels <- colnames(df)
  index1 <- which(all_labels==var1)
  index2 <- which(all_labels==var2)
  
  z <- table(df[,index1],df[,index2])
  return(chisq.test(z,correct=FALSE)$p.value)
}

for(i in 1:max){
  var1 <- categorical_variables[i]
  for(j in 1:max){
    var2 <- categorical_variables[j]
    all_p_values[i,j] <- return_chi2_test(training_set,var1,var2)
  }
}
all_p_values

# Train random forest model

threshold_selection <- function(model_optimal,validation_set){
  prediction <- predict(model_optimal,data = 
                          validation_set[,1:length(colnames(validation_set))-1])$predictions
  threshold <- seq(0.1,0.6,by=0.05)
  misclasserror <- rep(NA,length(threshold))
  
  for(i in 1:length(threshold)){
    
    misclasserror[i] <- misClassError(validation_set$income, prediction, threshold =threshold[i])
  }
  print(misclasserror)
  return(threshold[which.min(misclasserror)])
}

# A grid of tuning parameters
tuneGrid <- expand.grid(mtry = c(4,5,6,7,8,9,10,11,12),max.depth=c(3,8,15,30))
measure <- rep(NA,nrow(tuneGrid))

for(i in 1:nrow(tuneGrid)){
  model_ranger <- ranger(income ~ .,data = training_set,importance = "impurity"
                         ,mtry=tuneGrid[i,]$mtry,max.depth= tuneGrid[i,]$max.depth)
  
  prediction_temp <- predict(model_ranger,data = 
                        validation_set[,1:length(colnames(validation_set))-1])$predictions
  test_ROC <- roc(validation_set$income, prediction_temp,levels = c(0, 1), direction = "<")
  measure[i] <- auc(test_ROC)
}

# Fit the best model on the entire training dataset
optimal_parameters <- tuneGrid[which.max(measure),]
optimal_model <- ranger(income ~ .,data = training_set,importance = "impurity"
                        ,mtry=optimal_parameters$mtry,max.depth= optimal_parameters$max.depth)
# Determine the threshold using validation dataset
optimal_threshold <- threshold_selection(optimal_model,validation_set)

# Performance of random forest on test dataset
prediction_ranger <- predict(optimal_model,data = test_set)$predictions

obtain_metric_forest <- function(test_set, prediction_ranger, optimal_threshold){
  
  test_ROC <- roc(test_set$income, prediction_ranger,levels = c(0, 1), direction = "<")
  plot(test_ROC, col = "red")
  auc_value <- as.numeric(auc(test_ROC))
  
  pred<-ifelse(prediction_ranger>optimal_threshold,1,0) 
  
  # test
  outcomes <- table(test_set$income, pred)
  confusion <- conf_mat(outcomes)
  metric<-summary(confusion, event_level = "second")$.estimate
  accuracy<-metric[1]
  sensitivity_value<-metric[3]
  specificity_value<-metric[4]
  
  newlist<-list(accuracy, sensitivity_value, specificity_value, auc_value)
  cat("this function returns accuracy, sensitivity, specificity and auc in order \n")
  return(newlist)
}

randomforest_metric<-obtain_metric_forest(test_set, prediction_ranger, optimal_threshold)
randomforest_metric

##################### Neural Networks: Do Not Run #####################

# New dataset for use
neural_data <- income_data

# Normalize data for the model
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

neural_data$age <- normalize(neural_data$age)
neural_data$fnlwgt <- normalize(neural_data$fnlwgt)
neural_data$educational_num <- normalize(neural_data$educational_num)
neural_data$capital_gain <- normalize(neural_data$capital_gain)
neural_data$capital_loss <- normalize(neural_data$capital_loss)
neural_data$hours_per_week <- normalize(neural_data$hours_per_week)
neural_data$country <- as.factor(neural_data$country)

# Converting Factor Variables
# Make dummy variables of 6 columns
neural_data <- dummy_cols(neural_data, 
                          select_columns = c('workclass', 'marital_status', "occupation",
                                             "relationship", "race", "gender", "country"))

# Leaving out unnecessary variables
neural_data <- subset(neural_data, select = - c(workclass, marital_status, occupation,
                                                relationship, race, gender, education, country))

# Make sure no special characters
names(neural_data) <- str_replace_all(names(neural_data),"-", "_")
names(neural_data) <- str_replace_all(names(neural_data),"\\.", "_")
names(neural_data) <- str_replace_all(names(neural_data)," ", "_")

# Checking if the only remaining variables are int & num
str(neural_data)

# Train & Test Split
train_neural <- neural_data[split,]
test_neural <- neural_data[-split,]

# Setting up the model with formula
name_f <- names(train_neural)
f <- paste("income ~", paste(name_f[!name_f %in% "income"], collapse = " + "))
f <- as.formula(f)

# Selected 7 neurons in the one hidden layer, consistent with the rules in NN
# Mean of input/Output layers as the number of neurons, and 1 hidden layer by default
neural_model <- neuralnet(formula = f, 
                          data = train_neural,
                          hidden = 7,
                          err.fct = "ce",
                          act.fct = "logistic",
                          linear.output = FALSE,
                          lifesign = 'full',
                          rep = 1,
                          algorithm = "rprop+",
                          stepmax = 5e5)

# plot our neural network 
plot(neural_model, rep = 1)

# Resulting error
neural_model$result.matrix

# Prediction
output <- compute(neural_model, rep = 1, test_neural[, !names(test_neural) %in% c("income")])

# Confusion Matrix -Test Data
p1 <- as.vector(output$net.result)
pred1 <- ifelse(p1 > 0.5, 1, 0)
tab1 <- table(test_neural$income,pred1)
tab1

# Further plotting, can be moved to evaluation section
confusion_neural <- conf_mat(tab1)

autoplot(confusion_neural)

# Get summary metrics
summary(confusion_neural, event_level = "second")

# Then we use the ROC/AUC

# Plot the ROC of the stepwise model
ROC_neural <- roc(test_neural$income, p1)
plot(ROC_neural, col = "red")
auc(ROC_neural)

neural_metric<-list(summary(confusion_neural, event_level = "second")$.estimate[1],
                    summary(confusion_neural, event_level = "second")$.estimate[3],
                    summary(confusion_neural, event_level = "second")$.estimate[4],
                    as.numeric(auc(ROC_neural)))
#neural_metric
######################## Model Evaluation ######################## 
first_column <- c("accuracy", "sensitivity", "specificity", "auc")

logistic_metric_column<-c(logistic_metric[[1]], logistic_metric[[2]], 
                          logistic_metric[[3]], logistic_metric[[4]])
naivebayes_metric_column<-c(naivebayes_metric[[1]], naivebayes_metric[[2]], 
                            naivebayes_metric[[3]], naivebayes_metric[[4]])
knn_metric_column<-c(knn_metric[[1]], knn_metric[[2]], 
                     knn_metric[[3]], knn_metric[[4]])
tree_metric_column<-c(tree_metric[[1]], tree_metric[[2]], 
                      tree_metric[[3]], tree_metric[[4]])
randomforest_metric_column<-c(randomforest_metric[[1]], randomforest_metric[[2]], 
                              randomforest_metric[[3]], randomforest_metric[[4]])
svm_metric_column<-c(svm_metric[[1]], svm_metric[[2]], 
                     svm_metric[[3]], svm_metric[[4]])
#neural_metric_column<-c(neural_metric[[1]], neural_metric[[2]], 
#                        neural_metric[[3]], neural_metric[[4]])

metric_df<-data.frame(first_column, logistic_metric_column,naivebayes_metric_column,
                      knn_metric_column,tree_metric_column,
                      randomforest_metric_column, svm_metric_column) #neural_metric_column

metric_df
