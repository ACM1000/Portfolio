############ Integrantes del grupo: Alejandro Cañadas, Antonio Morcillo y Geovanny Jaramillo############



#Variable target: "Loan Status"// Variables Categoricas(factores): "loan_intent","loan_home_ownership" // Variables numéricas: 7 
library(tidymodels)
library(dplyr)
getwd()
setwd("C:/Users/Alex/Desktop/Loan trabajo/Intento1")
loan_data <- read.csv("credit_risk_dataset.csv")
head(loan_data)     #Primeras filas
summary(loan_data)  #Estadísticas descriptivas
str(loan_data)      #Tipo de variable
dim(loan_data)      #Dimension
names(loan_data)    #Nombres de las variables
##------------------------------------------------------------------------------------------------------------
## Muchos Outliers a la derecha, claramente asimetrica, sesgo hacia valores menores: 

plot1 <- ggplot(loan_data, aes(x = loan_amnt)) + 
  geom_histogram(bins=50, col= "White") 
plot2 <- ggplot(loan_data, aes(y = loan_amnt)) + 
  geom_boxplot()+coord_flip()
gridExtra::grid.arrange(plot1, plot2)

## Con logaritmos en loan_amount

loan_data$loan_amnt_log <- log10(loan_data$loan_amnt)

plot1 <- ggplot(loan_data, aes(x = loan_amnt_log)) + 
  geom_histogram(bins=50, col= "White") 
plot2 <- ggplot(loan_data, aes(y = loan_amnt_log)) + 
  geom_boxplot()+coord_flip()
gridExtra::grid.arrange(plot1, plot2)

## Raiz cuadrada

loan_data$loan_amnt_sqrt <- sqrt(loan_data$loan_amnt)

plot1 <- ggplot(loan_data, aes(x = loan_amnt_sqrt)) + 
  geom_histogram(bins=50, col= "White") 
plot2 <- ggplot(loan_data, aes(y = loan_amnt_sqrt)) + 
  geom_boxplot()+coord_flip()
gridExtra::grid.arrange(plot1, plot2)

## Transformacion menos agresiva
loan_data$loan_amnt_cbrt <- loan_data$loan_amnt^(1/3)

plot1 <- ggplot(loan_data, aes(x = loan_amnt_cbrt)) + 
  geom_histogram(bins=50, col= "White") 
plot2 <- ggplot(loan_data, aes(y = loan_amnt_cbrt)) + 
  geom_boxplot()+coord_flip()
gridExtra::grid.arrange(plot1, plot2)

loan_data<-loan_data|>
  mutate(loan_amnt_cbrt=loan_data$loan_amnt^(1/3))
loan_data$loan_amnt_scaled <- NULL


# Analizamos laas estadísticas de loan_status 
loan_data |>
  count(loan_status) |>
  mutate(percentage = n / sum(n) * 100)

## en relación a loan_intent
loan_data_summary_intent <- loan_data |>
  group_by(loan_intent) |>
  summarise(
    total_loans = n(),
    approved_loans_intent = sum(loan_status == 1),
    rejected_loans_intent = sum(loan_status == 0),
    approval_rate_intent = mean(loan_status == 1),
    rejection_rate_intent = mean(loan_status == 0)
  );loan_data_summary_intent



# Eliminar valores nulos y sustituirlos por la mediana
sum(is.na(loan_data))
colSums(is.na(loan_data)) #Suma de valores nulos

loan_data[loan_data == ""] <- NA


loan_data$person_emp_length[is.na(loan_data$person_emp_length)] <- median(loan_data$person_emp_length, na.rm = TRUE)
loan_data$loan_int_rate[is.na(loan_data$loan_int_rate)] <- median(loan_data$loan_int_rate, na.rm = TRUE)

sum(is.na(loan_data$person_emp_length)) #Comprobación
sum(is.na(loan_data$loan_int_rate))


## Feature engineering - Codificacion de los features cualitativos y ordenar factores ordinales:I,II,III, nominales: A,B,C,D
# Obtener los nombres de las variables categóricas
nominal_variables <- loan_data |> 
  select(where(is.factor))  # Selecciona columnas categóricas (factores)

# Verificar el número de niveles y la suma de cada nivel
lapply(nominal_variables, table)  

## Convertir variables categóricas en factores

nominal_features <- c('person_home_ownership','cb_person_default_on_file','loan_intent')
ordinal_features <- c('loan_grade')
numeric_predictors <- loan_data |> select(where(is.numeric))

loan_data$person_home_ownership <- as.factor(loan_data$person_home_ownership)
loan_data$cb_person_default_on_file <- as.factor(loan_data$cb_person_default_on_file)
loan_data$loan_intent <- as.factor(loan_data$loan_intent)
loan_data$loan_status <- as.factor(loan_data$loan_status)

## 'loan_grade' a factor ordenado
loan_data <- loan_data|>
  mutate(loan_grade = factor(loan_grade, levels =c('A','B','C','D','E','F','G'),ordered = TRUE))

contr_one_hot(levels(loan_data$person_home_ownership))
contr.poly(levels(loan_data$loan_grade))

loan_data <- loan_data |>
  mutate(
    person_home_ownership = as.factor(person_home_ownership),
    loan_intent = as.factor(loan_intent),
    loan_grade = factor(loan_grade, ordered = TRUE),  # Convertir a ordered factor
    cb_person_default_on_file = as.factor(cb_person_default_on_file),
    loan_status = as.factor(loan_status)) 

## Data Splitting

library(rsample)
set.seed(123)
loan_split <- initial_split(loan_data, prop = 0.7)
loan_train <- training(loan_split)
loan_test <- testing(loan_split)
table1 <- table(loan_data$loan_status);table1
table2 <- table(loan_train$loan_status);table2
table3 <- table(loan_test$loan_status);table3             

##-------------------------------PASO 2 ----------------------------------------------
loan_recipe <- recipe(loan_status~.,data=loan_data)|>
  step_nzv(all_predictors())|>
  step_naomit(all_predictors()) |>
  step_integer(all_ordered_predictors()) |>
  step_other(all_nominal_predictors(), threshold = 0.01) |>
  step_impute_median(all_numeric_predictors())|>
  #Imputamos con la mediana los valores que faltan
  step_ordinalscore(all_ordered_predictors())|>
  #Como no tenemos ninguna variables con demasiados niveles no usamos step_other para agrupar
  step_dummy(all_nominal_predictors(),one_hot=FALSE)|>
  step_YeoJohnson(all_numeric_predictors())|>
  step_normalize(all_numeric_predictors())
#al no haber altas correlaciones no aplicamos PCA   

# Aplicando recipe
loan_recipe_prepared <- prep(loan_recipe, training = loan_data)
transformed_data <- bake(loan_recipe_prepared, new_data = NULL)  

#Comprobando que la recipe ha funcionado
head (transformed_data) #Valores normalizados
str(transformed_data) #Solo valores numericos 
sum(is.na(transformed_data))
summary(transformed_data)

# Calculo de matriz de correlación
numeric_vars <- loan_data |> dplyr::select(where(is.numeric))
 cor_matrix <- cor(numeric_vars, use = "complete.obs") # Excluye valores NA
print(cor_matrix)

plot1 <- ggplot(transformed_data, aes(x = loan_amnt)) + 
  geom_histogram(bins=50, col= "White") 
plot2 <- ggplot(transformed_data, aes(y = loan_amnt)) + 
  geom_boxplot()+coord_flip()
gridExtra::grid.arrange(plot1, plot2)

##----------------------------------------PASO 3-----------------------------------------
## Creamos modelos y workflow 

ridge_model <- logistic_reg(penalty=tune(),mixture=0)|>
  set_engine("glmnet")|>
  set_mode("classification")
lasso_model <- logistic_reg(penalty = tune(), mixture = 1) |> 
  set_engine("glmnet")|>
  set_mode("classification")
elastic_net_model <- logistic_reg(penalty = tune(), mixture = tune()) |> 
  set_engine("glmnet")|>
  set_mode("classification")

# Creamos workflows
ridge_workflow <- workflow()|>
  add_model(ridge_model)|>
  add_recipe(loan_recipe)

lasso_workflow <- workflow() |> 
  add_model(lasso_model) |> 
  add_recipe(loan_recipe)

# Workflow para Elastic Net
elastic_net_workflow <- workflow() |> 
  add_model(elastic_net_model) |> 
  add_recipe(loan_recipe)

#Generamos los grids para tunear lambda (la penalización)

grid_ridge_tune <- grid_max_entropy(
  extract_parameter_set_dials(ridge_workflow),
  size = 50)

grid_lasso_tune <- grid_max_entropy(
  extract_parameter_set_dials(lasso_workflow),
  size = 50)

grid_elastic_net_tune <- grid_max_entropy(
  extract_parameter_set_dials(elastic_net_workflow),
  size = 50)

##--------------------------------PASO 4------------------------------------------
#Generamos folds
set.seed(123)  
loan_cvrep <- rsample::vfold_cv(data = loan_train, 
                                v = 5, 
                                repeats = 5, 
                                strata = loan_status)
ctrl <- control_resamples(event_level = "second")
metricas <- metric_set(roc_auc,f_meas,accuracy, precision, sens, spec)

# Ajuste de hiperparametros 
ridge_reg_tune <-tune_grid(ridge_workflow, 
                           loan_cvrep, 
                           grid = grid_ridge_tune, 
                           metrics = metricas,
                           control = ctrl)

lasso_reg_tune <-tune_grid(lasso_workflow, 
                           loan_cvrep, 
                           grid = grid_lasso_tune, 
                           metrics = metricas,
                           control = ctrl)

elastic_net_tune <- tune_grid(elastic_net_workflow,
                              loan_cvrep,
                              grid = grid_elastic_net_tune,
                              metrics = metricas,
                              control = ctrl)

##seleccionar el mejor valor de landa segun el roc_auc

best_ridge <- select_best(ridge_reg_tune, metric = "roc_auc")
best_lasso <- select_best(lasso_reg_tune, metric = "roc_auc")
best_elastic_net <- select_best(elastic_net_tune, metric = "roc_auc")

ridge_metrics <- ridge_reg_tune |> collect_metrics();print(ridge_metrics)
lasso_metrics <- lasso_reg_tune |> collect_metrics();print(lasso_metrics)
elastic_net_metrics <- elastic_net_tune |> collect_metrics();print(elastic_net_metrics)

## Metricas con mejor penalizacion , mejor valor roc_auc
ridge_full_metrics <- ridge_metrics |> 
  filter(penalty == best_ridge$penalty)
print(ridge_full_metrics)

lasso_full_metrics <- lasso_metrics |> 
  filter(penalty == best_lasso$penalty)
print(lasso_full_metrics)

elastic_net_full_metrics <- elastic_net_metrics |> 
  filter(penalty == best_elastic_net$penalty & mixture == best_elastic_net$mixture)
print(elastic_net_full_metrics)

#Se finaliza el modelo con los mejores hiperparámetros

final_ridge_wf <- finalize_workflow(ridge_workflow,best_ridge);final_ridge_wf
final_lasso_wf <- finalize_workflow(lasso_workflow, best_lasso);final_lasso_wf
final_elastic_wf <- finalize_workflow(elastic_net_workflow, best_elastic_net);final_elastic_wf

# Entrena el modelo final
final_ridge_fit <- fit(final_ridge_wf, data = loan_train)
final_lasso_fit <- fit(final_lasso_wf, data = loan_train)
final_elastic_fit <- fit(final_elastic_wf, data = loan_train)

## Da los mismos valores paso 4
##--------------------------------------PASO 5 ----------------------------------------------------

set.seed(123)
loan_cv_final <- rsample::vfold_cv(
  data = loan_train, v = 5, repeats= 10,
  strata = loan_status)

ctrl <- control_resamples(event_level = "second")
metricas <- metric_set(roc_auc, f_meas, accuracy, precision, sens, spec)


ridge_assessment <- fit_resamples(
  final_ridge_wf,
  resamples = loan_cv_final,
  metrics = metricas,
  control = ctrl)

lasso_assessment <- fit_resamples(
  final_lasso_wf,
  resamples = loan_cv_final,
  metrics = metricas,
  control = ctrl)


elastic_assessment <- fit_resamples(
  final_elastic_wf,
  resamples = loan_cv_final,
  metrics = metricas,
  control = ctrl)

ridge_metrics_asses1 <- collect_metrics(ridge_assessment)
lasso_metrics_asses1 <- collect_metrics(lasso_assessment)
elastic_metrics_asses1 <- collect_metrics(elastic_assessment)


ridge_metrics_asses1
lasso_metrics_asses1
elastic_metrics_asses1

## Metricas muy similares


#-----------------------------------PASO 6--------------------------------------------

#Por las metricas vemos que el mejor modelo es elastic-net por lo que trabajamos con el
## (Modelo train ya entrenado en el paso 4)

#A continuación tenemos las predicciones sobre el conjunto prueba 
# Este código genera predicciones soft (type="prob") para el modelo final, genera probabilidades predichas de pertenencia a cada clase,representan las probabilidades para las clases "No" (0) y "Yes" (1)
# De esta forma no genera la prediccion final pero si nos permite ajustar el ubral de probabilidad
ridge_test_preds <- predict(final_ridge_fit, loan_test, type = "prob");ridge_test_preds
lasso_test_preds <- predict(final_lasso_fit, loan_test, type = "prob");lasso_test_preds
elastic_test_preds <- predict(final_elastic_fit, loan_test, type = "prob");elastic_test_preds

#Para establecer el umbral queremos saber el numero de casos positivo que hay en la columna loan_status
prop_positivos <- mean(loan_train$loan_status == 1)
print(prop_positivos)

# Lo que nos da 0.2168 por tanto hay cerca de 22% positivos de default y 78% negativos  en el test
# Usamos esa porporcion como umbral
best_threshold <- 0.2155573
#Añadimos columna como factor renombrando como 1 y 0.
elastic_test_preds <- elastic_test_preds |> 
  mutate(preds = if_else(.pred_1 > best_threshold, 1, 0)) |> 
  mutate(preds = factor(preds, levels = c(0, 1), labels = c("No", "Yes")))

loan_test <- loan_test |> 
  mutate(loan_status = factor(loan_status, levels = c(0, 1), labels = c("No", "Yes")))
# loan_status en loan_test tiene los mismos niveles
# Cargar métricas a usar
metricas <- metric_set(accuracy, f_meas, roc_auc, precision, sens, spec)


elastic_test_metrics <- elastic_test_preds |> 
  bind_cols(truth = loan_test$loan_status) |> # Agregar la columna `truth` de los valores reales
  metricas(truth = truth, estimate = preds, .pred_1, event_level = "second") # Calcular métricas

# Mostrar métricas finales
print(elastic_test_metrics)




#-----------------------------------------------------------------------------------

#### --------------------------  Modelo no lineal , BOOSTING 
#####roc, acuracidad, sensibilidad, especificidad =1 
library(xgboost)

boost_model <- boost_tree(trees = 500, 
                          tree_depth = 1,2,4,
                          learn_rate = 0.001,0.01,0.1,
                          loss_reduction = 0.01,
                          min_n = 20) |> 
  set_engine("xgboost") |> 
  set_mode("classification")

# Workflow boosting
loan_workflow <- workflow() |>
  add_recipe(loan_recipe) |>  
  add_model(boost_model)  

# Entrenar el modelo 
boost_fit <- loan_workflow |>
  fit(data = loan_train)

boost_fit

## Log loss de 0,68 a 0.22. el modelo fue entrenado correctamente con 500 iteraciones utilizando 19 características. 
##La pérdida (logloss) disminuyó significativamente, lo que indica que el modelo se ajustó bien a los datos de entrenamiento.

########### Tuneando 

# Definir el modelo de Boosting con la sintonización de parámetros
boost_model_tune <- boost_tree(
  trees = tune(), 
  tree_depth = tune(), 
  min_n = tune(), 
  learn_rate = tune()
) |>
  set_engine("xgboost") |>
  set_mode("classification")


wflow_boost_tune <- workflow() |>
  add_model(boost_model_tune) |>
  add_recipe(loan_recipe)  

# Extraer el conjunto de parámetros para sintonizar
wflow_boost_tune_param  <- extract_parameter_set_dials(wflow_boost_tune)


grid_boost_tune <- expand.grid(
  trees = c(1000, 2000),  # Número de árboles
  tree_depth = c(1, 2, 4), 
  min_n = c(20, 50),    # Observaciones por nodo
  learn_rate = c(0.001, 0.01, 0.1))

print(grid_boost_tune)



####-------------------------------
library(parallel)
library(doParallel)

set.seed(123)
k1 <- 5; k2 <- 1
attr_cvrep <- rsample::vfold_cv(loan_train, v = k1, repeats = k2)
ctrl <- control_grid(event_level = "second") 
metricas <- metric_set(accuracy, sens, precision, f_meas, spec, roc_auc, detection_prevalence)
cl <- makePSOCKcluster(6)
registerDoParallel(cl)
boost_reg_tune <- tune_grid(wflow_boost_tune,
                            attr_cvrep,
                            grid = grid_boost_tune,
                            metrics = metricas,
                            control = ctrl)
stopCluster(cl)
boost_reg_tune |> collect_metrics()

best_params_boost <- select_best(boost_reg_tune, metric = "roc_auc")  
print(best_params_boost)

# Filtrar las métricas para la configuración óptima

best_config <- best_params_boost$.config

# Filtrar métricas del objeto `svm_tune`
best_boost_metrics <- boost_reg_tune|>
  collect_metrics() |>
  filter(.config == best_config)

best_boost_metrics

####-------------------------------------------------
## Definimos modelo con mejores hiperparámetros y lo integramos al workflow

final_boost_spec <- boost_tree(
  trees = best_params_boost$trees,
  min_n = best_params_boost$min_n,
  tree_depth = best_params_boost$tree_depth,
  learn_rate = best_params_boost$learn_rate
) |>
  set_engine("xgboost") |>
  set_mode("classification")

# Crear el workflow con el mejor modelo
final_boost_workflow <- workflow() |>
  add_model(final_boost_spec) |>
  add_recipe(loan_recipe)

library(yardstick)

# Entrenar el modelo con el conjunto de entrenamiento
final_boost_fit <- final_boost_workflow |>
  fit(data = loan_train)

#######################    Predicciones del modelo  ########################################


# Hacer predicciones
predictions <- predict(final_boost_fit, new_data = loan_test, type = "class")

# Convertir predicciones a "Yes" y "No"
predictions <- predictions |>
  mutate(.pred_class = ifelse(.pred_class == 1, "Yes", "No")) |>
  mutate(.pred_class = factor(.pred_class, levels = c("No", "Yes")))

# Crear dataframe con predicciones y valores reales
results <- loan_test |>
  select(loan_status) |>
  bind_cols(predictions)

metrics <- metric_set(accuracy, sens, spec, precision, f_meas, detection_prevalence)
# Calcular métricas
model_metrics <- results |>
  metrics(truth = loan_status, estimate = .pred_class,event_level = "second")

# Mostrar las métricas
print(model_metrics)

################################################-------------------------------------VIP
if (!requireNamespace("vip", quietly = TRUE)) {
  install.packages("vip")
}
library(vip)
library(xgboost)

best_params <- grid_boost_tune[1, ]

boost_model <- boost_tree(
  trees = best_params$trees,
  tree_depth = best_params$tree_depth,
  min_n = best_params$min_n,
  learn_rate = best_params$learn_rate
) |> 
  set_engine("xgboost", colsample_bytree = 1) |> 
  set_mode("classification")

boost_fit <- boost_model |> 
  fit(loan_status ~ ., data = loan_train)

importance_matrix <- xgboost::xgb.importance(model = extract_fit_engine(boost_fit))
print(importance_matrix)

vip::vip(extract_fit_engine(boost_fit), num_features = max(20, nrow(importance_matrix))) +  
  ggtitle("Importancia de Variables - XGBoost") +
  theme_minimal()



### Modelo SVM ---

set.seed(123)
loan_cvrp <- rsample::vfold_cv(loan_train, v = 5,  repeats = 1)

svmlin_loan_model <- svm_linear(cost = 5) |> 
  set_engine("kernlab")  |>
  set_mode("classification")
svmlin_wf <- workflow() |>
  add_recipe(loan_recipe) |> 
  add_model(svmlin_loan_model)
ctrl <- control_resamples(event_level = "second")
perf_metr_loan <- metric_set(accuracy, detection_prevalence, sens, precision, f_meas, spec, roc_auc)
asses1_loan_cvrp <- fit_resamples(svmlin_wf,
                                  resamples = loan_cvrp,
                                  metrics = perf_metr_loan,
                                  control = ctrl)
collect_metrics(asses1_loan_cvrp)
#--------------------------------Modelo radial-----------------------------------------------
svmrb_model <- svm_rbf(cost = 1, rbf_sigma = 0.01) |> 
  set_engine("kernlab")  |>
  set_mode("classification")
svmrb_wf <- workflow() |> 
  add_recipe(loan_recipe) |> 
  add_model(svmrb_model)
asses3_loan_cvrp <- fit_resamples(svmrb_wf,
                                  resamples = loan_cvrp,
                                  metrics = perf_metr_loan,
                                  control = ctrl)
collect_metrics(asses3_loan_cvrp)

#-----------------------------Modelo tuneado

# Generamos grid
svmrb_model_tune <- svm_rbf(cost = tune(), rbf_sigma = tune()) |> 
  set_engine("kernlab")  |>
  set_mode("classification")
svmrb_wf_tune <- workflow() |> add_recipe(loan_recipe) |> 
  add_model(svmrb_model_tune)
svmrb_wf_tune_param  <- extract_parameter_set_dials(svmrb_wf_tune)
grid_svmrb_wf_tune <- expand.grid(cost = cost() |> 
                                    value_seq(n = 5, original = TRUE), 
                                  rbf_sigma = rbf_sigma() |> 
                                    value_seq(n = 5, original = TRUE))

cost()
rbf_sigma()
grid_svmrb_wf_tune
## ------------------------------------------------------------------------

library(tidymodels)
library(doParallel)

# Configurar paralelización
cl <- makePSOCKcluster(parallel::detectCores() - 2)
registerDoParallel(cl)

# Configurar control para tuning
ctrl_tune <- control_grid(save_pred = TRUE, verbose = TRUE)

# Ejecutar tuning con validación cruzada
svmrb_tune_results <- tune_grid(
  svmrb_wf_tune,
  resamples = loan_cvrp,
  grid = grid_svmrb_wf_tune,
  metrics = perf_metr_loan,
  control = ctrl_tune
)
stopCluster(cl)

svmrb_tune_results |>
  collect_metrics()

#----------------------------------------------

best_params_svm <- select_best(svmrb_tune_results, metric = "roc_auc")  
print(best_params_svm)

# Filtrar las métricas según los mejores hiperparámetros de roc_auc
best_svm_metrics <- svmrb_tune_results |>
  collect_metrics() |>
  filter(cost == best_params_svm$cost, rbf_sigma == best_params_svm$rbf_sigma)

# Mostrar los resultados
print(best_svm_metrics)

####-------------------------------------------------
## Definimos modelo con mejores hiperparámetros y lo integramos al workflow

final_svm_model <- svm_rbf(cost = best_params_svm$cost, 
                           rbf_sigma = best_params_svm$rbf_sigma) |>
  set_engine("kernlab") |>
  set_mode("classification")

# Crear el workflow con el mejor modelo
final_svm_wf <- workflow() |>
  add_recipe(loan_recipe) |>
  add_model(final_svm_model)

# Y entrenamos el modelo con los datos del entrenamiento

final_svm_fit <- final_svm_wf |>
  fit(data = loan_train)

##Hacemos predicciones sobre conjunto de prueba
svm_predictions <- final_svm_fit |>
  predict(new_data = loan_test) |>
  bind_cols(loan_test) 

# Convertir predicciones de 0/1 a "No"/"Yes"
svm_predictions <- svm_predictions |>
  mutate(
    .pred_class = ifelse(.pred_class == 1, "Yes", "No"),  # Transformar valores
    .pred_class = factor(.pred_class, levels = c("No", "Yes"))  # Convertir a factor con niveles correctos
  )

# Asegurar que la variable loan_status también tenga los mismos niveles
svm_predictions <- svm_predictions |>
  mutate(
    loan_status = factor(loan_status, levels = c("No", "Yes"))
  )

print(levels(svm_predictions$.pred_class))
print(levels(svm_predictions$loan_status))
# Definir las métricas que queremos calcular
metricas_completas <- metric_set(accuracy, sens, spec, precision, f_meas, detection_prevalence)

# Calcular todas las métricas usando la probabilidad de "Yes"
final_metrics <- svm_predictions |>
  metricas_completas(truth = loan_status, estimate = .pred_class, 
                     prob = .pred_Yes)  

# Mostrar métricas finales
print(final_metrics)








































