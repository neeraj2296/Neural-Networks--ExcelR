#Including the necassary packages
library(neuralnet)  # regression
library(nnet) # classification 
library(NeuralNetTools)
library(plyr)

#Loading the dataset
Startups <- read.csv(file.choose())
View(Startups)
class(Startups)
#Replacing the categories witha numerical values to factorise it.
Startups$State <- as.numeric(revalue(Startups$State,c("New York"="1", "California"="2","Florida"="3")))#Replacing the categories witha numerical values to factorise it.
str(Startups)
Startups <- as.data.frame(Startups)
attach(Startups)

# Exploratory data Analysis :
plot(R.D.Spend, Profit)
plot(Administration, Profit)
plot(Marketing.Spend, Profit)
plot(State, Profit)
windows()
# Find the correlation between Output (Profit) & inputs (R.D Spend, Administration, Marketing, State) - SCATTER DIAGRAM
pairs(Startups)

# Correlation coefficient - Strength & Direction of correlation
cor(Startups)

summary(Startups) # Confirms on the different scale and demands normalizing the data.

# Apply Normalization technique to the whole dataset :
#Normalize function is defined
normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}
Startups_norm<-as.data.frame(lapply(Startups,FUN=normalize))#dataset is normalised
summary(Startups_norm$Profit) # Normalized form of profit

summary(Startups$profit) # Orginal profit value

# Data is Partitioned as train and test set 
set.seed(123)
ind <- sample(2, nrow(Startups_norm), replace = TRUE, prob = c(0.7,0.3))
Startups_train <- Startups_norm[ind==1,]
startups_test  <- Startups_norm[ind==2,]

# Creating a neural network model on training data
startups_model <- neuralnet(Profit~R.D.Spend+Administration
                            +Marketing.Spend+State,data = Startups_train)
str(startups_model)

#Visualising the results from the model
plot(startups_model, rep = "best")
summary(startups_model)
par(mar = numeric(4), family = 'serif')
plotnet(startups_model, alpha = 0.6)


# Evaluating model performance
set.seed(12323)
model_results <- compute(startups_model,startups_test[1:4])
predicted_profit <- model_results$net.result
# Predicted profit Vs Actual profit of test data.
cor(predicted_profit,startups_test$Profit)


# since the prediction is in Normalized form, we need to de-normalize it 
# to get the actual prediction on profit
str_max <- max(Startups$Profit)
str_min <- min(Startups$Profit)

#Creating the unormaalize function to retrieve back the actual prediction data
unnormalize <- function(x, min, max) { 
  return( (max - min)*x + min )
}
ActualProfit_pred <- unnormalize(predicted_profit,str_min,str_max)
head(ActualProfit_pred)

# Improve the model performance :
set.seed(12345)
Startups_model2 <- neuralnet(Profit~R.D.Spend+Administration
                             +Marketing.Spend+State,data = Startups_train,
                             hidden = 2)
plot(Startups_model2 ,rep = "best")
summary(Startups_model2)

# Evaluating model performance
model_results2<-compute(Startups_model2,startups_test[1:4])
predicted_Profit2<-model_results2$net.result
cor(predicted_Profit2,startups_test$Profit)

#Visualising the results from the model
plot(predicted_Profit2,startups_test$Profit)
par(mar = numeric(4), family = 'serif')
plotnet(Startups_model2, alpha = 0.6)
# SSE(Error) has reduced and training steps had been increased as the number of neurons  under hidden layer are increased