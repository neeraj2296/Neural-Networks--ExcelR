#Including the necassary packages
library(neuralnet)  # regression
library(nnet) # classification 
library(NeuralNetTools)
library(plyr)

#Loading the dataset
forest <- read.csv(file.choose())
View(forest)
class(forest)
#Replacing the categories witha numerical values to factorise it.
forest$month <- as.integer(as.factor(forest$month))#Replacing the categories witha numerical values to factorise it.
forest$day <- as.integer(as.factor(forest$day))
forest$size_category <- as.integer(as.factor(forest$size_category))
str(forest)
forest <- as.data.frame(forest)
attach(forest)
colnames(forest)

# Exploratory data Analysis :
windows()
# Find the correlation between Output (area) & inputs (R.D Spend, Administration, Marketing, State) - SCATTER DIAGRAM
pairs(forest)

# Correlation coefficient - Strength & Direction of correlation
cor(forest)

summary(forest) # Confirms on the different scale and demands normalizing the data.

# Apply Normalization technique to the whole dataset :
#Normalize function is defined
normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}
forest_norm<-as.data.frame(lapply(forest,FUN=normalize))#dataset is normalised
summary(forest_norm$area) # Normalized form of area

summary(forest$area) # Orginal area value

# Data is Partitioned as train and test set 
set.seed(123)
ind <- sample(2, nrow(forest_norm), replace = TRUE, prob = c(0.7,0.3))
forest_train <- forest_norm[ind==1,]
forest_test  <- forest_norm[ind==2,]

# Prediction of Forest fires requires only prediction from 
# temperature, rain, relative humidity and wind speed
#Normalizing the following variables.
forest$temp = normalize(forest$temp)
forest$RH   = normalize(forest$RH)
forest$wind = normalize(forest$wind)
forest$rain = normalize(forest$rain)
# We need to tweak this as a classification problem.lets base out the Size using this criteria :
attach(forest)

# Creating a neural network model on training data
forest_model <- neuralnet(area~temp+wind+RH+rain,data = forest_train)
str(forest_model)

#Visualising the results from the model
plot(forest_model, rep = "best")
summary(forest_model)
par(mar = numeric(4), family = 'serif')
plotnet(forest_model, alpha = 0.6)


# Evaluating model performance
set.seed(12323)
model_results <- compute(forest_model,forest_test[1:4])
predicted_area <- model_results$net.result
# Predicted area Vs Actual area of test data.
cor(predicted_area,forest_test$area)


# since the prediction is in Normalized form, we need to de-normalize it 
# to get the actual prediction on area
ff_max <- max(forest$area)
ff_min <- min(forest$area)

#Creating the unormaalize function to retrieve back the actual prediction data
unnormalize <- function(x, min, max) { 
  return( (max - min)*x + min )
}
Actualarea_pred <- unnormalize(predicted_area,ff_min,ff_max)
head(Actualarea_pred)

# Improve the model performance :
set.seed(12345)
forest_model2 <- neuralnet(area~temp+wind+RH+rain,data = forest_train,hidden = 2)
plot(forest_model2 ,rep = "best")
summary(forest_model2)

# Evaluating model performance
model_results2<-compute(forest_model2,forest_test[1:4])
predicted_area2<-model_results2$net.result
cor(predicted_area2,forest_test$area)

#Visualising the results from the model
plot(predicted_area2,forest_test$area)
par(mar = numeric(4), family = 'serif')
plotnet(forest_model2, alpha = 0.6)
# SSE(Error) has reduced and training steps had been increased as the number of neurons  under hidden layer are increased