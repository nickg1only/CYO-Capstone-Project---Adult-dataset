options(echo = FALSE, verbose = FALSE, warn = FALSE)

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

# Train the models.


#Decision Tree:

train_rpart <- train(train_x, train_y, method = "rpart")
plot(train_rpart)

rpart_preds <- predict(train_rpart, test_x)
rpart_acc <- mean(rpart_preds == test_y)


#Random Forest - type and enter "Y" when asked to run the following model,
                # otherwise, hit enter or type something and hit enter):
run_rf <- readline(prompt = "Run Random Forest? (WARNING: VERY SLOW)")

if(run_rf == "Y"){
  train_rf <- train(train_x, train_y, method = "rf") #VERY SLOW
  plot(train_rf)
  
  rf_preds <- predict(train_rf, test_x)
  rf_acc <- mean(rf_preds == test_y)
  
  rm(train_rf, rf_preds)
}

rm(test_index, test_x, train_rpart, train_x, rf_preds, rpart_preds, test_y, train_y)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#RESULTS:


cat("Accuracies:")
cat("Decision Tree: ", rpart_acc)
if(run_rf == "Y") cat("Random Forest: ", rf_acc)
