options(warn = -1)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#DATA PREPARATION:


# The following code will create the training and test datasets:


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)


#Download the file containing the dataset:

dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", dl)

adult <- fread(text = readLines(dl),
               col.names = c("age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"))

rm(dl)


# Test set will be 10% of Adult data

set.seed(1, sample.kind = "Rounding")

test_index <- createDataPartition(y = adult$income, times = 1, p = 0.1, list = FALSE)
train <- adult[-test_index,]
test <- adult[test_index,]


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Separate the outcome from the features.

train_x <- train[,-15]
train_y <- as.factor(train$income)

test_x <- test[,-15]
test_y <- as.factor(test$income)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Introduce the project to the user:


cat("\n\n","Goal: Predict if income level is above 50K")
cat("\n","Dataset: Adults in the 1994 Census")
cat("\n\n","Features: ")
cat(names(train_x), sep = "\n\t")
cat("\n","Outcome: ","Income > 50K (Yes/No)")

cat("\n\n\nPredictive Models: ","\n","\tDecision Tree","\n","\tRandom Forest","\n")


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Train the models.



#Decision Tree:

train_rpart <- train(train_x, train_y, method = "rpart")
rpart_best_cp <- train_rpart$bestTune

rpart_preds <- predict(train_rpart, test_x)
rpart_acc <- mean(rpart_preds == test_y)



#Random Forest - 
# Enter "Y" when asked to run the following model,
# otherwise, enter "N" or type anything and hit enter):

run_rf <- readline(prompt = "Run Random Forest? (WARNING: VERY SLOW) Hit enter for Yes, type NO for No:  ")

if(run_rf == ""){
  train_rf <- train(train_x, train_y, method = "rf") #VERY SLOW
  rf_best_cp <- train_rf$bestTune
  
  rf_preds <- predict(train_rf, test_x)
  rf_acc <- mean(rf_preds == test_y)
  
}


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#RESULTS:

cat("\n\n-----\n\n","RESULTS: ","\n\n-------\n")

cat("Decision Tree: ","\n\n")
cat("Best Complexity Parameter: ", as.numeric(rpart_best_cp), "\n")
cat("Plot for Decision Tree: (see Plots)","\n\n")
plot(train_rpart)
cat("Accuracy: ", rpart_acc)

cat("\n\n\n ----- \n\n")

if(run_rf == ""){
  cat("Random Forest: ","\n\n")
  cat("Best Complexity Parameter: ", as.numeric(rf_best_cp), "\n")
  cat("Plot for Random Forest: (see Plots)","\n\n")
  plot(train_rf)
  cat("Accuracy: ", rf_acc)
}


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Remove unnecessary values:


rm(test_index, test_x, train_x, test_y, train_y)

rm(train_rpart, rpart_preds)

if(run_rf == ""){
  rm(train_rf, rf_preds)
}
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

options(warn = 0)