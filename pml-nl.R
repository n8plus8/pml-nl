# Practical Machine Learning
# by Johns Hopkins University
#  Course project
#  Weight Lifting Exercises Dataset

# ==============================================================================
# Import Data
# ==============================================================================

# install.packages(c("caret", "doSNOW"), dependencies = TRUE)
library(caret); # library(doSNOW) # for training in parallel, optional

training <- data.table::fread("pml-training.csv", stringsAsFactors = FALSE)
testing <- read.csv("pml-testing.csv", stringsAsFactors = FALSE)


# ==============================================================================
# Tidying
# ==============================================================================

View(training) # We see a lot of blanks and NAs

# "classe" is a factor
training$classe <- as.factor(training$classe)


# We can remove the index, names, and most likely the timestamps 
f_train <- data.frame(training[ , -(1:5)]) # f, for filtered


# Dealing with NAs
# Total NAs per column set to 95% threshold to keep columns where we may be able
# to impute missing values later
f_train <- f_train[ , colSums(!is.na(f_train)) > .95*nrow(f_train)]
sum(is.na(f_train)) # looks like all columns had over 95% NAs


# Removing columns with more than 95% blanks
empties <- apply(f_train, 2, function(x) sum(x == "")) > .95*nrow(f_train)
f_train <- f_train[ , !empties]
sum(f_train == "") # looks like all columns had over 95% blanks


# We could have probably eliminated problematic variables with nearZeroVar()
# Let's see what it tells us now
nzv <- nearZeroVar(f_train[,-55])
nzv # 1 = first variable, the "new_window" variable
table(f_train[,1]) # let's drop it
f_train <- f_train[ , -nzv]

table(training$classe) # class balance seems to be ok

# at this point we can continue to preprocess, such as removing correlated
# predictors -say if we wanted to build a partial least squares model
# or we can center and scale the variables for PCA, but we can use preProcess
# in the train function



# ==============================================================================
# Modeling
# ==============================================================================

# let's run some models
# CART	
# Flexible Discriminant Analysis (FDA)	
# Learning Vector Quantization (LVQ)	
# Linear Discriminant Analysis (LDA)	
# Quadratic Discriminant Analysis (QDA)	
# Random Forests (RF)	
# Regularized Discriminant Analysis (RDA)	
# Stochastic Gradient Boosting aka Gradient Boosted Machine (GBM)	using xgbTree
# Support Vector Machine (SVM) using svmRadial


set.seed(88888888)

control <- trainControl(method = "repeatedcv",
                        number = 10,
                        repeats = 3)

# Hyperparameter grid search for xgbTree
tune.grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)


cl <- makeCluster(4, type = "SOCK")
# Register cluster for parallel training in caret
registerDoSNOW(cl)


modelCART <- train(classe ~ ., data = f_train, method = "rpart",
                   trControl = control)

modelFDA <- train(classe ~ ., data = f_train, method = "fda",
                   trControl = control)

modelLVQ <- train(classe ~ ., data = f_train, method = "lvq",
                  trControl = control)

modelLDA <- train(classe ~ ., data = f_train, method = "lda",
                  trControl = control)

modelQDA <- train(classe ~ ., data = f_train, method = "qda",
                  trControl = control)

modelRF <- train(classe ~ ., data = f_train, method = "rf",
                  trControl = control)

modelRDA <- train(classe ~ ., data = f_train, method = "rda",
                  trControl = control)

modelGBM <- train(classe ~ ., data = f_train, method = "xgbTree",
                  trControl = control, tuneGrid = tune.grid, verbose = FALSE)

modelSVM <- train(classe ~ ., data = f_train, method = "svmRadial",
                  trControl = control)

stopCluster(cl)



results <- resamples(list(CART = modelCART,
                          FDA = modelFDA,
                          LVQ = modelLVQ,
                          LDA = modelLDA,
                          QDA = modelQDA,
                          RF = modelRF,
                          RDA = modelRDA,
                          GBM = modelGBM,
                          SVM = modelSVM))

summary(results)
bwplot(results)

confusionMatrix(training$classe, predict(modelSVM))$overall[1]
confusionMatrix(training$classe, predict(modelGBM))$overall[1]
confusionMatrix(training$classe, predict(modelBLR))$overall[1]
confusionMatrix(training$classe, predict(modelQDA))$overall[1]

predictions <- predict(modelGBM, testing)
predictions