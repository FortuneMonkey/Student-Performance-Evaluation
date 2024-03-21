#Library
library(readxl)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(magrittr)
library(plotrix)
library(polycor)
library(caret)
library(pROC)
library(xgboost)
library(naivebayes)
library(caTools)
library(klar)
library(e1071)
library(car)
library(RColorBrewer)
library(randomForest) 
library(ggcorrplot)


#Data Import
PATH="C:\\Users\\Saputra\\Student Prediction Dataset.csv"
data <- read.csv(PATH, sep = ",") #data is the variable for the excel file

#Findings
View(data)
View(new_Data)
summary(data)
colnames(data)
is.na(data)
anyNA(data)
table(data$NOTES)
table(data$GRADE)
str(data)

#Cleaning / Pre-Processing
sum(is.na(data))

anycolna <- sapply(data,anyNA)
anycolna

anydupl <- data %>%
  filter(duplicated(.))
anydupl

# The Evaluation of Students Grade Level
# NOTE: Always Refresh/Remove the Variables That has been Used

Grade_Data <- data %>%
  group_by(GRADE) %>%
  summarize(n = n()) %>%
  mutate(pct = n / sum(n),
         lbl = scales::percent(pct))

Grade_Data$GRADE <- factor(Grade_Data$GRADE, levels = c(0, 1, 2, 3, 4, 5, 6, 7), 
                           labels = c("Fail", "DD", "DC", "CC", "CB", "BB", "BA", "AA"), ordered = TRUE)

ggplot(Grade_Data, aes(x = GRADE,y=n, fill = GRADE)) +
  geom_bar(position = "dodge", colour = "black", stat = "identity") +  
  geom_text(aes(label = lbl), size = 3, position = position_dodge(width = 0.9), vjust = -0.5) +  
  labs(title = "Evaluation of Students Grade Level",
       x = "Grade",
       y = "Count") +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal()


#========================================================================================================================#
#Question 1: What is The Impact of Students Taking Notes during class? 

# Analysis 1-1: The distribution of students who taking notes during class (Univariate) 

x <- c(53, 587, 894)
piepercent<- round(x/sum(x)*100)
piepercent <- paste(piepercent ,"%")
labels <- c(piepercent)
pie(x, labels, main = "Pie Chart of Student Who Takes Notes", col = c("#A1FB8E","#FFFE91","#73FBFD"))
legend("topleft", c("Never","Sometimes","Always"), cex = 0.6,
       fill = c("#A1FB8E","#FFFE91","#73FBFD"))

# Analysis 1-2: Relationship between students taking notes and grade (Bivariate)
# NOTE: Always Refresh/Remove the Variables That has been Used

data$GRADE <- factor(data$GRADE, levels=c(0,1,2,3,4,5,6,7), labels = c("Fail", "DD", "DC", "CC", "CB", "BB", "BA", "AA"))
data$NOTES <- factor(data$NOTES, levels=c(1,2,3))

notes_grade<-ggplot(data,aes(x=NOTES,fill=NOTES))+
  geom_bar(position="dodge",color="black")+
  labs(title="Relationship Between Notes Frequency and Grades",
       x="Notes Frequency",
       y="Count",
       fill="Grade")+
  theme_minimal()+
  facet_wrap(~GRADE,scales="free",ncol=3)
print(notes_grade)

# Analysis 1-3: Relationship Between Students Taking Notes, Grade and Regular sports Activity (Multivariate)
# NOTE: Always Refresh/Remove the Variables That has been Used

data$GRADE<-as.factor(data$GRADE)
data$NOTES<-as.factor(data$NOTES)
data$ACTIVITY<-as.factor(data$ACTIVITY)

plotdata<-data%>%
  group_by(GRADE,NOTES,ACTIVITY)%>%
  summarize(n=n())%>%
  mutate(pct=n/sum(n),
         lbl=scales::percent(pct))

ggplot(plotdata,
       aes(x=factor(GRADE,levels = c("0","1","2","3","4","5","6","7"), 
                    labels = c("Fail", "DD", "DC", "CC", "CB", "BB", "BA", "AA")),
           y=pct,
           fill=factor(NOTES,levels=c("1","2","3"),
                       labels=c("Never","Sometimes","Always"))))+
  geom_bar(stat="identity",position="fill",width=0.7)+
  geom_text(aes(label=lbl),size=3,position=position_fill(vjust=0.5))+
  facet_grid(cols=vars(factor(ACTIVITY,levels=c("1","2"),labels=c("Yes","No"))))+
  scale_y_continuous(labels=scales::percent_format(scale=1),expand=c(0,0))+
  scale_fill_brewer(palette="Set2")+
  labs(y="Percent",fill="NOTES",x="GRADE",title="Student Performance by Taking Notes and Activities")+
  theme_minimal()+
  theme(axis.text.x=element_text(angle=0,hjust=1))

print(n=40,plotdata)
view(plotdata)

# Analysis 1-4: Evaluating the Association Between Students Taking Notes and Grade Using Chi-Square Analysis
contingency_table<-table(data$GRADE,data$NOTES)
chi_squared_result<-chisq.test(contingency_table)
print(chi_squared_result)

# Analysis 1-5: Correlation Between Students Taking Notes and Grade Using Polychoric
polychor(data$NOTES,data$GRADE)


#Machine Learning 1: Logistic Regression 

# Load necessary libraries
library(caret)
library(pROC)

# Select relevant columns and convert 'GRADE' to binary
lgmodel <- data %>%
  select(GRADE, NOTES, WORK, SCHOLARSHIP, ATTEND) %>%
  mutate(GRADE = as.numeric(GRADE %in% 5:7))

# Set seed for reproducibility
set.seed(1234)

# split data into training and testing sets
splitIndex <- createDataPartition(y = lgmodel$GRADE, p = 0.8, list = FALSE)
train_data <- lgmodel[splitIndex, ]
test_data <- lgmodel[-splitIndex, ]

# Fit Logistic Regression Model
model <- glm(GRADE ~ NOTES + WORK + SCHOLARSHIP + ATTEND, data = train_data, family = "binomial")

# Display a summary of the logistic regression model
summary(model)

# Make predictions on the test set
predictions <- predict(model, newdata = test_data, type = "response")

# Convert predicted probabilities to binary predictions
predictions_binary <- ifelse(predictions > 0.5, 1, 0)

# Display confusion matrix
conf_matrix <- confusionMatrix(as.factor(predictions_binary), as.factor(test_data$GRADE))
print(conf_matrix)

# Calculate and display accuracy
accuracy <- sum(predictions_binary == test_data$GRADE) / length(test_data$GRADE)
cat("Accuracy:", round(accuracy, 3), "\n")

# ROC Curve & Area under the curve
roc_obj <- roc(test_data$GRADE, predictions)

# Plot ROC curve with AUC
plot(roc_obj, col = "blue", main = "ROC Curve of Logistic Regression", col.main = "darkred", lwd = 3)
text(0.8, 1.0, paste("AUC =", round(auc(roc_obj), 4)), col = "darkred", cex = 1.2)

#========================================================================================================================#

#Question 2: What is the correlation between students who have and did not have additional work with their grade in class?

#Analysis 2-1: Univariate analysis on students having additional work and their grades 
#Univariate bar chart count
ggplot(unvplot, aes(x = WORK, y = n)) + 
  geom_bar(stat="identity", fill='forestgreen', col='black') +
  geom_text(aes(label = n), vjust=-0.5) +
  labs(x = "Category", 
       y = "Frequency", 
       title  = "Student with Additional Work Distribution")

#Analysis 2-1: Univariate analysis on students having additional work and their grades 
#Univariate bar chart percentage
unv_plot_percentage <- data %>%
  count(WORK) %>%
  mutate(pct = n / sum(n),
         pctlabel = paste0(round(pct*100), "%"))

ggplot(unv_plot_percentage, 
       aes(x = reorder(WORK, pct), y = pct)) + 
  geom_bar(stat="identity", fill="forestgreen", color="black") +
  geom_text(aes(label = pctlabel), vjust=-0.25) +
  scale_y_continuous(labels = percent) +
  labs(x = "Work", 
       y = "Percent", 
       title  = "Student with Additional Work")

#Analysis 2-2: Bivariate analysis on students having additional work and their grades.
#bivariate plot
biv_plot <- xtabs(~WORK+GRADE, data=data)
plot(biv_plot, main="Student with Additional Work and Their Grade", col="forestgreen")

#Analysis 2-2: Bivariate analysis on students having additional work and their grades.
#bivariate bar chart count
brv_bar_count <- data %>%
  group_by(WORK, GRADE) %>%
  summarise(Count = n())

ggplot(brv_bar_count, aes(x = WORK, y = Count, fill = GRADE)) + 
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Count),
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
  ylab("Count") +
  labs(title = "Count of Students by Additional Work and Grade") +
  theme_minimal()

#Analysis 2-2: Bivariate analysis on students having additional work and their grades.
#Bivariate bar chart percentage
brv_bar_pct <- data %>%
  group_by(WORK, GRADE) %>%
  summarise(Count = n()) %>%
  group_by(WORK) %>%
  mutate(Percentage = Count / sum(Count) * 100)

ggplot(brv_bar_pct, aes(x = WORK, y = Percentage, fill = GRADE)) + 
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = sprintf("%0.1f%%", Percentage)),
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
  labs(y = "Percentage",
       x = "Category",
       title = "Percentage of Students by Additional Work and Grade") +
  scale_y_continuous(labels = scales::percent_format(scale = 1)) +
  theme_minimal()

#Analysis 2-3: Multivariate analysis on student’s grade and their frequency of taking notes grouped by their additional work.
#Multivariate bar chart count
ggplot(data, aes(x = NOTES, fill = GRADE)) + 
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = ..count..),
            position = position_dodge(width = 0.9), vjust = -0.5) +
  facet_wrap(~WORK) +
  xlab("NOTES") +
  ylab("Count") +
  labs(title = "Count of Student's Grade and their Frequency of Taking Notes Grouped by their Additional Work") +
  theme_bw()

#Analysis 2-3: Multivariate analysis on student’s grade and their frequency of taking notes grouped by their additional work.
#multivariate bar chart percentage
multivariate_bar <- data %>%
  group_by(WORK, NOTES, GRADE) %>%
  summarise(Count = n()) %>%
  group_by(WORK, NOTES) %>%
  mutate(Percentage = Count / sum(Count) * 100)

ggplot(multivariate_bar, aes(x = NOTES, y = Percentage, fill = GRADE)) + 
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = sprintf("%0.1f%%", Percentage)),
            position = position_dodge(width = 0.9), vjust = -0.5) +
  facet_wrap(~WORK) +
  xlab("NOTES") +
  ylab("Percentage") +
  scale_y_continuous(labels = scales::percent_format(scale = 1)) +  
  theme_bw()

# Analysis 2-4: chi square test  
contingency_table<-table(data$GRADE,data$WORK)
chi_squared_result<-chisq.test(contingency_table)
print(chi_squared_result)

# Analysis 2-5: polychor test
polychor(data$GRADE,data$WORK)

# Machine Learning 2: Extreme Gradient Boosting Machine Learning 
dataset = df[, c("SCHOLARSHIP", "WORK", "ATTEND", "NOTES", "GRADE")]
data = data.frame(dataset)
head(data)

# Load the necessary packages
library(xgboost)
library(caret)

# Split the data into training and testing sets
set.seed(123)  # Set seed for reproducibility
split_index <- createDataPartition(data$GRADE, p = 0.7, list = FALSE)
train_data <- data[split_index, ]
test_data <- data[-split_index, ]

# Define the X and Y variables
X_train <- subset(train_data, select = -GRADE) 
y_train <- train_data$GRADE
X_test <- subset(test_data, select = -GRADE)
y_test <- test_data$GRADE

# Convert data to matrix format for XGBoost --> data transformation
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

# Specify XGBoost parameters
params <- list(
  objective = "multi:softmax",  # for multi-class classification
  num_class = length(unique(data$GRADE)),  # number of classes
  eval_metric = "merror"  # log-likelihood loss for multi-class
)

# Train the XGBoost model
model <- xgboost(
  params = params,
  data = dtrain,
  nrounds = 50,  
  verbose = 1  
)

# Make predictions on the test set
predictions <- predict(model, newdata = as.matrix(X_test))
predictions = as.factor(predictions)
y_test = as.factor(y_test)
all_levels <- levels(y_test)
predictions <- factor(predictions, levels = all_levels)

levels(predictions)
levels(y_test)

# Create confusion matrix
conf_matrix <- confusionMatrix(predictions, y_test)

# Print the confusion matrix
print(conf_matrix)

# Convert confusion matrix to a data frame
conf_matrix_df <- as.data.frame(as.table(conf_matrix))

# Plot with ggplot2
ggplot(conf_matrix_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradientn(colors = c("white", "forestgreen")) +
  theme_bw() +
  labs(title = "Confusion Matrix",
       x = "Actual",
       y = "Predicted")

# Access feature importance values
importance_values <- xgb.importance(model = model)

# Print feature importance
print(importance_values)

# Plot feature importance
xgb.plot.importance(importance_matrix = importance_values)

#========================================================================================================================#

#Question 3: What is The Relation and Impact between Students who get scholarship with their Grade? 

data$SCHOLARSHIP <- factor(data$SCHOLARSHIP, levels = c(1,2,3,4,5), labels=c("None", "25%", "50%", "75%", "Full"), ordered = TRUE)
data$GRADE <- factor(data$GRADE, levels = c(0,1,2,3,4,5,6,7), labels=c("Fail", "DD", "DC", "CC","CB","BB","BA", "AA"), ordered=TRUE)
data$GENDER <- factor(data$GENDER, levels = c(1,2), labels=c("Female","Male"), ordered=TRUE)

#Analysis 3-1: The Distribution of Students who Has Scholarship 
ggplot(data, aes(x = SCHOLARSHIP, fill = SCHOLARSHIP)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..), vjust=-1) +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Barplot of Scholarship", x = "Scholarship", y = "Count")

#Analysis 3-2: Relationship Between Students Scholarship with their Grade 
data %>%
  mutate(GRADE = factor(GRADE, levels = c("AA", "BA", "BB", "CB", "CC","DC","DD","Fail"))) %>%
  group_by(SCHOLARSHIP, GRADE) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n)) %>%
  ggplot(aes(x = SCHOLARSHIP, y = n, fill = GRADE)) +
  geom_bar(stat = "identity", position = "stack") +
  geom_text(aes(label = scales::percent(freq)), position = position_stack(vjust = 0.5), size = 3) +
  labs(x = "SCHOLARSHIP", y = "GRADE") +
  scale_fill_brewer(palette = "Set3") +
  theme_minimal()

#Analysis 3-3: Relationship between Grade, Scholarship in accordance with Gender 
ggplot(data, aes(y = GENDER, fill = SCHOLARSHIP)) +
  geom_bar(position = "dodge", size = 1) +  
  geom_text(stat = "count", aes(label = ..count..),
            position = position_dodge(width = 1), vjust = 0.5, size = 3) +
  facet_wrap(~GRADE) +
  xlab("GRADE") +
  ylab("GENDER") +
  scale_fill_brewer(palette = "Set3") +
  guides(fill = guide_legend(title = "SCHOLARSHIP")) +
  theme_bw() 

#Analysis 3-4: Using Chi-Square Analysis to Determine the Relation between Grade and Scholarship 
tbl <- table(data$SCHOLARSHIP, data$GRADE)
chisq.test(tbl)

#Analysis 3-5: Correlation between Students who get Scholarship with their Grade with Polychor 
polychor(data$SCHOLARSHIP,data$GRADE)

# Machine Learning 3: RFA Random Fortress Approach 

df = read.csv(PATH, header = TRUE) [-1]
data = data.frame(df)
head(data)


# Load the necessary packages
library(randomForest) 
library(caret)


# Split the data into training and testing sets
set.seed(123)  # Set seed for reproducibility
split_index <- createDataPartition(data$GRADE, p = 0.8, list = FALSE)
train_data <- data[split_index, ]
test_data <- data[-split_index, ]

# Define the X and Y variables
X_train <- subset(train_data, select = -GRADE) 
y_train <- train_data$GRADE
X_test <- subset(test_data, select = -GRADE)
y_test <- test_data$GRADE


# Ensure consistent levels for the target variable
levels_union <- union(levels(as.factor(y_train)), levels(as.factor(y_test)))

# Set the levels for y_train and y_test
y_train <- factor(y_train, levels = levels_union)
y_test <- factor(y_test, levels = levels_union)

# Train the Random Forest model
rf_model <- randomForest(x = X_train, y = y_train, ntree = 500)

# Make predictions on the test set
predictions <- predict(rf_model, newdata = X_test)

# Display the confusion matrix for Random Forest
conf_matrix_rf <- confusionMatrix(predictions, y_test)
print(conf_matrix_rf)
#========================================================================================================================#

#Question 4:What is the impact of students who always attend class between their grades?

#factor
data$ATTEND <- factor(data$ATTEND,levels = c(1,2,3), labels =c("always","sometimes","never"))
data$NOTES <- factor(data$NOTES,levels = c(1,2,3), labels =c("never","sometimes","always"))
data$GRADE <- factor(data$GRADE,levels = c(0,1,2,3,4,5,6,7), labels =c("Fail","DD", "DC","CC","CB", "BB","BA", "AA"), ordered = TRUE)
data$WORK <- factor(data$WORK,levels = c(1,2), labels=c("Yes","No"))
data$LIKES_DISCUSS <- factor(data$LIKES_DISCUSS, levels = c(1,2,3), labels=c("never","sometimes","always"))

#Analysis 4-1:The distribution of students who always attending class(Univariate)
barplot(table(data$ATTEND), col = c("#00FF00", "#FFFF00", "#0000FF"),
        main = "Barplot of Attending Class", xlab = "Attendance",
        ylab = "Amount of Student", ylim = c(0, 1500), legend = TRUE)
custom_x <- barplot(contingency_attendance, plot = FALSE)
text(custom_x, contingency_attendance + 2, labels = contingency_attendance, pos = 3, col = "#0000FF",cex=0.8)

#Analysis 4-2:Relationship between attendance of student and grade (Bivariate)
barplot(table(data$GRADE, data$ATTEND), beside = TRUE, 
        col = c("#87CEEB", "#FFA500", "#008000", "#FFFF00", "#FF0000", "#800080", "#A52A2A", "#808080"), 
        main = "Grades by Attendance", xlab = "Attendance",
        ylab = "Amount of Student", ylim = c(0, 500), legend = TRUE)

#Analysis 4-3:Relationship between students who attending class, grade and notes (Multivariate)
plotdata<-data%>%
  group_by (GRADE, ATTEND, NOTES)%>%
  summarize (n=n ())%>%
  mutate (pct=n/sum(n),
          lbl=scales::percent (pct))

ggplot(plotdata,
       aes(x=GRADE,
           y=pct,
           fill=NOTES))+
  geom_bar(stat="identity", position="fill",width=0.7)+
  geom_text(aes(label=lbl),size=3, position= position_fill(vjust=0.5))+
  facet_grid(cols=vars(ATTEND))+
  scale_y_continuous(labels=scales::percent_format(scale=1), expand=c(0,0))+
  scale_fill_brewer(palette="Set2")+
  labs(y="Percent",fill="NOTES",x="GRADE",title="grade by attendance and taking note")+
  theme_minimal()+
  theme(axis.text.x=element_text(angle=0,hjust=1))

#Analysis 4-4:Evaluating the association between students attending class and grade using chi-square analysis 

#chi-square test
heh_contingency_table<- table(as.numeric(data$GRADE),as.numeric(data$ATTEND))
heh_chi_squared_result<-chisq.test(heh_contingency_table)
print(heh_chi_squared_result)

#Analysis 4-5:Correlation between students attending class and grade using polychoric

#calculate polychoric correlation
polychor(data$ATTEND,data$GRADE)

# Machine Learning 4: Naive Bayes

#Encoding the target variable
set.seed(123)
hehData <- data[, c("GRADE", "ATTEND", "LIKES_DISCUSS", "WORK", "SCHOLARSHIP")]
hehData <- hehData %>%
  mutate(GRADE = recode(GRADE, '0' = "Fail",
                        '1' = "Fail",
                        '2' = "Fail",
                        '3' = "Pass",
                        '4' = "Pass",
                        '5' = "Pass",
                        '6' = "Pass",
                        '7' = "Pass")) %>% 
  mutate(ATTEND = recode(ATTEND, '1' ="always",
                         '2' ="sometimes")) %>% 
  mutate(LIKES_DISCUSS = recode(LIKES_DISCUSS, '1' = "no",
                                '2' = "yes",
                                '3' = "yes")) %>%
  mutate(WORK = recode(WORK, '1'= "yes",
                       '2'= "no")) %>%
  mutate(SCHOLARSHIP = recode(SCHOLARSHIP, '1'= "less than 50%",
                              '2'= "less than 50%",
                              '3'= "more than 50%",
                              '4'= "more than 50%",
                              '5'= "more than 50%")) 

#training and test set
hehsample <- sample(c(TRUE, FALSE), nrow(hehData), replace = TRUE, prob = c(0.7, 0.3))
trainData <- hehData[hehsample, ]
testData <- hehData[-hehsample, ]

#Create naive bayes
heh_model <- naiveBayes(GRADE~ ATTEND+WORK+LIKES_DISCUSS+SCHOLARSHIP, data=trainData)
heh_model

# Predicting the test set output
heh_pred<-predict(heh_model,testData)
heh_pred

# Creating a Confusion Matrix
conf_matrx <- table(testData$GRADE, heh_pred)
conf_matrx

heh.accuracy_test <- sum(diag(conf_matrx)) / sum(conf_matrx)
print(paste('Accuracy for test', conf_matrx))

#Plot ROC curve for naive bayesian
roc_curve <- roc(as.numeric(testData$GRADE), as.numeric(heh_pred))
plot(roc_curve, main = "Naive Bayes - ROC", col = "#FF1493")
#Add AUC to the plot
text(0.5, 0.3, paste("AUC =", round(auc(roc_curve), 4)), col = "#0437F2", cex = 1.2)

#========================================================================================================================#


