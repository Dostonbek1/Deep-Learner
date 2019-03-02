library(recipes)


recep<-function(data_set){

recepie_obj <- recipe(data_set[0] ~ ., data=data_set)%>%
  step_dummy(all_predictors(),-all_outcomes())%>%
  step_knnimpute(all_predictors(),-all_outcomes())%>%
  step_center(all_predictors(),-all_outcomes())%>%
  step_scale(all_predictors(),-all_outcomes())%>%
  prep(data=train_tbl)
x_train<-bake(recepie_obj, new_data=train_tbl)
write.csv(as.matrix(x_train),file = "data/data_ready.csv")
}