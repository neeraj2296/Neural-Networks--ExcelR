#Including the necassary libraries
library(neuralnet)  # regression
library(nnet) # classification 
library(NeuralNetTools)

# Read the data
concrete <- read.csv(file.choose())
str(concrete)

# Exploratory data Analysis :
hist(concrete$cement, prob = T, breaks = 30)
lines(density(concrete$cement))
summary(concrete$cement)
hist(concrete$slag, prob = T, breaks = 30)
lines(density(concrete$slag))

summary(concrete$slag)

# Similar Histogram for eight other features confirms that the 
# data has different scales and needs a normalization.
summary(concrete) # Confirms on the different scale and demands normalizing the data.



# Apply Normalization technique to the whole dataset :
normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}
concrete_norm<-as.data.frame(lapply(concrete,FUN=normalize))
summary(concrete_norm$strength) # Normalized form of strength
summary(concrete$strength) # Orginal strength value


# Data is Partitioned as train and test set 
set.seed(123)
ind <- sample(2, nrow(concrete_norm), replace = TRUE, prob = c(0.7,0.3))
concrete_train <- concrete_norm[ind==1,]
concrete_test  <- concrete_norm[ind==2,]


# Creating a neural network model on training data

concrete_model <- neuralnet(strength~cement+slag+ash+water+superplastic+coarseagg+fineagg+age,data = concrete_train)
str(concrete_model)

plot(concrete_model, rep = "best")
summary(concrete_model)
par(mar = numeric(4), family = 'serif')
plotnet(concrete_model, alpha = 0.6)

# Evaluating model performance
set.seed(12323)
model_results <- compute(concrete_model,concrete_test[1:8])
predicted_strength <- model_results$net.result

# Predicted strength Vs Actual Strength of test data.
cor(predicted_strength,concrete_test$strength)

# since the prediction is in Normalized form, we need to de-normalize it 
# to get the actual prediction on strength.
str_max <- max(concrete$strength)
str_min <- min(concrete$strength)

unnormalize <- function(x, min, max) { 
  return( (max - min)*x + min )
}

ActualStrength_pred <- unnormalize(predicted_strength,str_min,str_max)
head(ActualStrength_pred)


# Improve the model performance :
set.seed(12345)
concrete_model2 <- neuralnet(strength~cement+slag+ash+water+superplastic+
                               coarseagg+fineagg+age,data= concrete_train,
                             hidden = 5)
plot(concrete_model2, rep = "best")
summary(concrete_model2)

#Evaluating the model 
model_results2<-compute(concrete_model2,concrete_test[1:8])

#Predicting using the bes model
predicted_strength2<-model_results2$net.result
cor(predicted_strength2,concrete_test$strength)

#Visualising thee predictions
plot(predicted_strength,concrete_test$strength)
par(mar = numeric(4), family = 'serif')
plotnet(concrete_model2, alpha = 0.6)
# SSE(Error) has reduced and training steps had been increased as the number of neurons # under hidden layer are increased
