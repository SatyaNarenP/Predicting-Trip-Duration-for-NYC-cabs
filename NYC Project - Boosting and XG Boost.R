rm(list = ls())
setwd("D:/Summer Semester/Intro to Predictive Modelling/Project")


## Reading the train file
filename = 'Cleaned Train.csv'
train_raw = read.csv(filename, header = TRUE, nrows = 100000)
summary(train_raw)

## Changing the month field as factor. Removing the trip_id column

train_raw$pickup_month = as.factor(train_raw$pickup_month)
summary(train_raw)
train_raw = train_raw[,-1]


## One hot encoding
library(data.table)
library(mltools)
encoded_train = one_hot(as.data.table(train_raw))
names(encoded_train)


## Creating Train and Test (validation) splits

set.seed(11)
train_rows = sample(1:nrow(encoded_train),(nrow(encoded_train)*0.8))

train_data = encoded_train[train_rows,]
train_data = na.omit(train_data)


test_data = encoded_train[-train_rows,]
test_data = na.omit(test_data)

train_x = train_data[,-2]
train_y = train_data[,2]
test_x = test_data[,-2]
test_y = test_data[,2]


## Implementing Boosting 

library(gbm)
set.seed(1)
boost.nyc = gbm(trip_duration~.,data = train_data, distribution = "gaussian",n.trees = 1000,
                shrinkage = 0.3,interaction.depth = 6,cv.folds = 3)


summary(boost.nyc)


yhat.boost = predict(boost.nyc,newdata = test_data,n.trees = 1000)

## rss prediction
sqrt(mean((yhat.boost - test_data$trip_duration)^2))
#RSS is 88645
rmse = sqrt(95049)
rmse




test_y_mean = mean(test_y$trip_duration,na.rm = TRUE)
# Calculate total sum of squares
tss =  sum((test_y$trip_duration - test_y_mean)^2 )
# Calculate residual sum of squares
rss =  sum((yhat.boost - test_data$trip_duration)^2)
# Calculate R-squared
rsq  =  1 - (rss/tss)
cat('The R-square of the test data is ', round(rsq,3), '\n')




## Implementing XG Boost

library(xgboost)

dtrain = xgb.DMatrix(data = as.matrix(train_data[,-2]),label = train_data$trip_duration)
dtest = xgb.DMatrix(data = as.matrix(test_data[,-2]),label = test_data$trip_duration)

params <- list(booster = "gbtree", objective = "reg:linear", eta=0.1, 
               gamma=0, max_depth=8, min_child_weight=1, 
               subsample=0.6, colsample_bytree=0.5)


## Trying to loop the parameters

eta_grid = c(0.1,0.3)
depth_grid = c(4,6,8)
nround_grid = c(500,1000,3000)




readings = data.frame(eta=double(0),depth=double(0),nround=double(0),rss = double(0))

iter = 1

for (i in 1:2){
  for (j in 1:3){
    for (k in 1:3){
      
      print ('eta considered is')
      print (i)
      print ('depth considered is')
      print (j)
      print ('ntrees considered is')
      print (k)
      parameters <- list(booster = "gbtree", objective = "reg:linear", eta=eta_grid[i], 
                         gamma=0, depth=depth_grid[j], min_child_weight=1, 
                         subsample=0.6, colsample_bytree=0.5)
      
      xgb_fit = xgb.train(params = parameters,data = dtrain,early_stopping_rounds = 20
                          ,watchlist = list(val=dtest,train=dtrain),nrounds = nround_grid[k])
      
      xgbpredict = predict(xgb_fit,dtest)
      rss = mean((xgbpredict-test_y$trip_duration)^2)
      
      readings[iter,1] = eta_grid[i]
      readings[iter,2] = depth_grid[j]
      readings[iter,3] = nround_grid[k]
      readings[iter,4] = rss
      iter = iter + 1
      
    }
  }
}

## Fitting the best model observed using hyper parameter tuning above


dtrain = xgb.DMatrix(data = as.matrix(train_data[,-2]),label = train_data$trip_duration)
dtest = xgb.DMatrix(data = as.matrix(test_data[,-2]),label = test_data$trip_duration)


params <- list(booster = "gbtree", objective = "reg:linear", eta=0.1, 
               gamma=0, max_depth=8, min_child_weight=1, 
               subsample=0.6, colsample_bytree=0.5)

xgb1 = xgb.train(params = params,data = dtrain,nrounds = 3000,early_stopping_rounds = 20
                 ,watchlist = list(val=dtest,train=dtrain))

xgbpredict = predict(xgb1,dtest)

##RSS of XGBoost
mean((xgbpredict-test_y$trip_duration)^2)
##RSS is 72547
RMSE = sqrt(mean((xgbpredict-test_y$trip_duration)^2))


## Calculating R square

test_y_mean = mean(test_y$trip_duration,na.rm = TRUE)
# Calculate total sum of squares
tss =  sum((test_y$trip_duration - test_y_mean)^2 )
# Calculate residual sum of squares
rss =  sum((xgbpredict-test_y$trip_duration)^2)
# Calculate R-squared
rsq  =  1 - (rss/tss)
cat('The R-square of the test data is ', round(rsq,3), '\n')


importance = xgb.importance(model = xgb1)


