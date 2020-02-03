rm(list = ls())
setwd("D:/Summer Semester/Intro to Predictive Modelling/Project")


## Reading the train file
filename = 'Cleaned Train.csv'
train_raw = read.csv(filename, header = TRUE)
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
boost.nyc = gbm(trip_duration~.,data = train_data, distribution = "gaussian",n.trees = 300,
                shrinkage = 0.3,interaction.depth = 6,cv.folds = 3)


summary(boost.nyc)


yhat.boost = predict(boost.nyc,newdata = test_data,n.trees = 300)

## rss prediction
mean((yhat.boost - test_data$trip_duration)^2)
#RSS is 88645
rmse = sqrt(88645)

## rmsle calculation
yy = yhat.boost
yy = ifelse(yy < 0 ,0,yy)
rmsle = sqrt(mean((log(yy+1) - log(test_data$trip_duration +1))^2))
rmsle
## RMSLE is 0.41  



## Implementing XG Boost

library(xgboost)


dtrain = xgb.DMatrix(data = as.matrix(train_data[,-2]),label = train_data$trip_duration)
dtest = xgb.DMatrix(data = as.matrix(test_data[,-2]),label = test_data$trip_duration)


params <- list(booster = "gbtree", objective = "reg:linear", eta=0.1, 
               gamma=0, max_depth=8, min_child_weight=1, 
               subsample=0.6, colsample_bytree=0.5)

#xgbcv = xgb.cv(params = params,data = dtrain,nrounds = 200,nfold = 5,showsd = T,
#               early.stopping.round = 20)
#min(xgbcv$evaluation_log[,4])


xgb1 = xgb.train(params = params,data = dtrain,nrounds = 3000,early_stopping_rounds = 20
                 ,watchlist = list(val=dtest,train=dtrain))

xgbpredict = predict(xgb1,dtest)

##RSS of XGBoost
mean((xgbpredict-test_y$trip_duration)^2)
##RSS is 72547

## rmsle calculation
rmsle = sqrt(mean((log(xgbpredict+1) - log(test_y$trip_duration +1))^2))
xx = xgbpredict
xx = ifelse(xx <0 , 0, xx)
rmsle = sqrt(mean((log(xx+1) - log(test_y$trip_duration +1))^2))
rmsle
## RMSLE is 0.386




test_y_mean = mean(test_y$trip_duration,na.rm = TRUE)
# Calculate total sum of squares
tss =  sum((test_y$trip_duration - test_y_mean)^2 )
# Calculate residual sum of squares
rss =  sum((xgbpredict-test_y$trip_duration)^2)
# Calculate R-squared
rsq  =  1 - (rss/tss)
cat('The R-square of the test data is ', round(rsq,3), '\n')






## Trying to loop the parameters

eta_grid = c(0.1,0.15,0.2,0.25,0.3,0.4)
depth_grid = c(4,5,6)
nround_grid = c(500,700,900,1000)




readings = data.frame(eta=double(0),depth=double(0),nround=double(0),rss = double(0))

iter = 1

for (i in 1:6){
  for (j in 1:3){
    for (k in 1:4){
      
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



#writing the readings as a dataframe

write.csv(readings,file = "XGBoost_readings.csv")

## Variable importance
xgb.importance(model = xgb1)



##################################################
#################################################
##BEST XG BOOST FIT

dtrain_best = xgb.DMatrix(data = as.matrix(train_data[,-c(2)]),label = train_data$trip_duration)
dtest_best = xgb.DMatrix(data = as.matrix(test_data[,-c(2)]),label = test_data$trip_duration)


#names(train_data[,-c(1,2,3:8,19:24,26,27,28:43)])

#names(train_data[,c(2)])


params_best <- list(booster = "gbtree", objective = "reg:linear", eta=0.1, 
               gamma=0, max_depth=8, min_child_weight=1, 
               subsample=0.6, colsample_bytree=0.5)

#xgbcv = xgb.cv(params = params,data = dtrain_best,nrounds = 1000,nfold = 5,showsd = T,
#               early.stopping.round = 20)
#min(xgbcv$evaluation_log[,4])

set.seed(1)


xgb_best = xgb.train(params = params_best,data = dtrain_best,nrounds = 1000,early_stopping_rounds = 20
                 ,watchlist = list(val=dtest_best,train=dtrain_best))

##xgb_boost = xgboost(params = params_best,data = dtrain_best,nrounds = 1000,early_stopping_rounds = 20)

xgbpredict = predict(xgb_best,dtest_best)

##RSS of XGBoost
mean((xgbpredict-test_y$trip_duration)^2)
##RSS is 75736

## rmsle calculation
rmsle = sqrt(mean((log(xgbpredict+1) - log(test_y$trip_duration +1))^2))
xx = xgbpredict
xx = ifelse(xx <0 , 0, xx)
rmsle = sqrt(mean((log(xx+1) - log(test_y$trip_duration +1))^2))
rmsle
## RMSLE is 0.39

xgb.importance(model = xgb_best)


## Fitting on the entire data

encoded_train = na.omit(encoded_train)
dtrain_total = xgb.DMatrix(data = as.matrix(encoded_train[,-c(1,2,3:8,19:24,26,27,28:43)]),label = encoded_train$trip_duration)
xgb_boost = xgboost(params = params_best,data = dtrain_total,nrounds = 1000,early_stopping_rounds = 20)

xgbpredict_total = predict(xgb_boost,dtest_best)
mean((xgbpredict_total-test_y$trip_duration)^2)




## Reading the test data
filename = 'Cleaned Test.csv'
test_raw = read.csv(filename, header = TRUE)
summary(test_raw)

test_raw$pickup_month = as.factor(test_raw$pickup_month)

## One hot encoding
library(data.table)
library(mltools)
test_new= test_raw[,-1]


encoded_test = one_hot(as.data.table(test_new))
names(encoded_test)

#names(encoded_test[,-c(1,2,3:7,18:23,25,26,27:42)])

xgbpredict_test = predict(xgb_boost,as.matrix(encoded_test[,-c(1,2,3:7,18:23,25,26,27:42)]))

predicted_result= data.frame(id=test_raw$id,trip_duration=xgbpredict_test)
head(predicted_result)

write.csv(predicted_result,file = "NYC_predictions.csv")


