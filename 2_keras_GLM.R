### ASTIN Data analytics workshop
# Script 2 - GLM in Keras

### define NN
k_clear_session()
input_list = list()
embedding_list = list()
cardinalities = list()
count = 0

### loop over categorical. for each, define an input layer and an output layer

### here we set embed dim to 1

for (column in paste0(cat_vars, "_input")){
  count = count + 1
  cardinalities[[column]] = train[, max(get(column))]
  print(cardinalities[[column]])
  input_list[[count]] = layer_input(shape = 1, dtype = "int32", name = column)
  embedding_list[[count]] = input_list[[count]] %>%
    layer_embedding(input_dim = cardinalities[[column]]+1, output_dim =1) %>% 
    layer_flatten(name = paste0(column, "_embed"))
}

### loop over continuous, for each, define an input layer and an output layer

for (column in paste0(cont_vars, "_input")){
  count = count + 1
  input_list[[count]] = layer_input(shape = 1, dtype = "float32", name = column)
  embedding_list[[count]] = input_list[[count]]
}

input_list[[count + 1]] = layer_input(shape = 1, dtype = "float32", name = "Exposure")

### join all embeddings together, then flatten to a vector and apply batch_norm to regularize

embeds = embedding_list %>% 
  layer_concatenate(axis = 1)

### rest of the network

middle = embeds %>% 
  layer_dense(units = 1, activation = "exponential")

### multiply the output of middle layer == frequency with exposure

output = list(middle,input_list[[count + 1]])  %>% layer_multiply() 

### define model

model = keras_model(inputs = c(input_list),
                    outputs = c(output))

###

model

### high learning rate

adam = optimizer_adam(learning_rate = 1)

model %>% compile(optimizer = adam, loss = "poisson")

model_write = callback_model_checkpoint(paste0("c:/r/astin_frmtpl_glm.h5"), save_best_only = T, verbose = 1)
learn_rate = callback_reduce_lr_on_plateau(factor = 0.9,patience = 5,cooldown = 0, verbose = 1)

## fit model. These are all good presents we can discuss.
 fit = fit(model, x = train_x, y = train_y, batch_size = 4096, epochs=50, callbacks=list(model_write, learn_rate),
           validation_split = 0.05, verbose = 1)

model = load_model_hdf5("c:/r/astin_frmtpl_glm.h5")

all[, pred_GLM := model %>% predict(all_x, batch_size = 32000)]
train[, pred_GLM :=  model %>% predict(train_x, batch_size = 32000)]
test[, pred_GLM :=  model %>% predict(test_x, batch_size = 32000)]
