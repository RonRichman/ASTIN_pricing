### ASTIN Data analytics workshop
# Script 1 - clean data

library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(keras)
library(tensorflow)
library(stringr)
library(lubridate)

### read in FRMTPL data

load("c:/r/freMTPL2freq_fixed.rda")

freMTPL2freq = freMTPL2freq %>% data.table()
freMTPL2freq[, id:= 1:.N]

freMTPL2freq[, Density := log(Density)]

### correct for data errors

freMTPL2freq[Exposure>1, Exposure:= 1]
freMTPL2freq[ClaimNb>=4, ClaimNb:= 4]

### define categorical versus continuous variables
var_names = freMTPL2freq %>% names()

cat_vars = var_names[c(3,9,8,11)]

cont_vars = var_names[c(4,5,6,7,10)]

### split train and test
set.seed(123)

RNGversion("3.5.0") # we use R version 3.5.0 for this partition
set.seed(500)
dat = freMTPL2freq %>% data.frame()
ll <- sample(c(1:nrow(dat)), round(0.9*nrow(dat)), replace = FALSE)
train <- dat[ll,] %>% data.table()
test <- dat[-ll,]%>% data.table()

train[, set := "train"]
test[, set := "test"]
dat = rbind(train, test)

### check empirical proportions

dat[, .(freq = sum(ClaimNb)/sum(Exposure)), keyby = .(set)]

### scale continuous and convert categorical to integers

min_max = function(col){
  dat = data.table(col_name = col)
  list(min = dat[, min(col_name)], max = dat[, max(col_name)], 
       scaled = dat[, (col_name - min(col_name))/(max(col_name) - min(col_name))])
}

for (col in cont_vars){
  mins_maxs = train[, get(col)] %>% min_max()
  train[, paste0(col, "_input") := mins_maxs$scaled]
  test[, paste0(col, "_input") := (get(col) - mins_maxs$min)/(mins_maxs$max - mins_maxs$min)]
}

cat_to_int = function(col){
  dat = data.table(col_name = col)
  dat = dat[order(col_name)] %>% unique()
  dat[, paste0("col_name", "_input") := as.integer(as.factor(col_name))]
  list(map = dat)
}

for (col in cat_vars){
  maps = (train[, get(col)] %>% cat_to_int())$map
  maps %>% setnames(names(maps), c(col, paste0(col, "_input")))
  train %>% setkeyv(col)
  train = train %>% merge(maps)
  test %>% setkeyv(col)
  test = test %>% merge(maps)
}

### get data in format for keras
get_keras_data = function(dat){
  temp = list()
  for (col in c(cat_vars, cont_vars)) temp[[paste0(col, "_input")]] = dat[, get(paste0(col, "_input"))] %>% as.matrix
  temp[["Exposure"]]  = dat[, get("Exposure")] %>% as.matrix
  temp
}

train_x = train %>% get_keras_data
test_x = test %>% get_keras_data

all = rbind(train, test)
all_x = all %>% get_keras_data()

train_y = train$ClaimNb %>% as.matrix
test_y = test$ClaimNb %>% as.matrix