# (0) Prelúdio ==========================================================

setwd('/home/heitor/Área de Trabalho/R Projects/Análise Macro/Lab 9')
library(tidyverse)
# library(neuralnet)  #!!: está com funções desativadas
library(tidymodels)
library(nnet)
library(NeuralNetTools)
library(devtools)
source_url('https://gist.github.com/fawda123/7471137/raw/cd6e6a0b0bdb4e065c597e52165e5ac887f5fe95/nnet_plot_update.r')
                        # p/ usar plot.nnet()
#library(keras)         # Ñ usarei esse:
#library(tensorflow)    #      tem algum problema no meu python
#library(tfruns)

# (1) Importação e Divisão ==============================================

dds <- read_csv("concrete.csv")

#normalize <- function(x) {  return( (x - min(x)) / (max(x) - min(x)))  }
#dds_norm  <- as.data.frame(lapply(dds, normalize))

# como será uma regressão (resposta quantitativa) e minhas variáveis são todas numéricas, não colocarei 'strata'

slice_1 <- initial_split(dds)
train   <- training(slice_1)
test    <- testing(slice_1)

# (2) Modelo ============================================================

nnet_1 <- mlp(
	hidden_units = integer(tune()),
	penalty      = double(tune()),
	activation   = 'linear') %>% 
	set_mode("regression") %>% 
	set_engine("nnet", verbose = 0) 
    # traduzir para pô-lo em termos de nnet:
nnet_1 %>% translate()

# (3) Tratamentos e Fórmula para Alimentar o Modelo =====================

recipe_1 <- recipe(strength~.,
				   data = train) %>% 
	step_normalize(all_numeric_predictors()) %>% 
	prep()

recipe_1 %>% bake(new_data=NULL)

# (4) Workflow ==========================================================

nnet_1_wrkflw <- workflow() %>%
	add_model(nnet_1) %>%
	add_recipe(recipe_1)

# (5) Validação Cruzada =================================================

valid_1 <- vfold_cv(train, v = 5)
					
# (6) Treinamento =======================================================

nnet_1_trained <- nnet_1_wrkflw %>% 
	tune_grid(valid_1,
			  grid    = 15,
			  control = control_grid(save_pred = T),
			  metrics = metric_set(ccc, mae)) 

nnet_1_trained %>% show_best(n=15)

# (6.1) Auto-plot ---
ggplot2::autoplot(nnet_1_trained)

# (6.2) Selecionando o melhor conjunto de parâmetros ---
best_tune  <- select_best(nnet_1_trained, 'ccc')
nnet_final <- nnet_1 %>%
	finalize_model(best_tune)

nnet_final %>% translate(engine = 'nnet') 

# (7) Testando ==========================================================

nnet_final_wrkflw <- workflow() %>% 
	add_recipe(recipe_1) %>% 
	add_model(nnet_final) %>% 
	last_fit(slice_1) %>% 
	collect_predictions()
nnet_final_wrkflw

nnet_final_wrkflw %>% 
	select(.row, .pred, strength) %>% 
	ggplot() +
	aes(x = strength,
		y = .pred) +
	geom_point() +
	geom_abline(intercept = 0,
				slope     = 1,
				color     ='red',
				size      = .8)

cor(nnet_final_wrkflw$.pred, nnet_final_wrkflw$strength)

# (A1) traduzir nnet_final para os termos de nnet pra fazer o gráfico:

nnet_final %>% translate()

nnet3 <-nnet(strength ~ cement+slag+ash+water+superplastic+coarseagg+fineagg+age,
			 size = 8, # as penalty
			 data = recipe(strength~., data = train) %>% 
			 	step_normalize(all_numeric_predictors()) %>% 
			 	prep() %>% bake(new_data=NULL),
			 decay = 0.0218099079606513,
			 verbose = 0, trace = FALSE, 
			 linout = TRUE)
nnet3 %>% plotnet()


