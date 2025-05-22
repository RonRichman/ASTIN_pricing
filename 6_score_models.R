
#### Score model

score_model = function(train, test, pred_col) {

train[, pred := train[, get(pred_col)]]
test[, pred := test[, get(pred_col)]]

poisson_deviance =  function(true, pred) {2*(sum(pred)- sum(true) + sum (log((true/pred)^(true))))/length(true)}

pois_dev = poisson_deviance(train$ClaimNb, train$pred)
pois_dev_test = poisson_deviance(test$ClaimNb, test$pred)

results = data.table(pois_dev = pois_dev,
                     pois_dev_test= pois_dev_test,
                     model = pred_col)

return(list(results = results))

}

glm_scores = score_model(train, test, "pred_GLM")
NN_scores = score_model(train, test, "pred_NN")
NN_NE_scores = score_model(train, test, "pred_NN_NE")
NN_NE_cann_scores = score_model(train, test, "pred_NN_NE_cann")

all_results = rbind(glm_scores$results,NN_scores$results, NN_NE_scores$results,NN_NE_cann_scores$results)

### check results
test = test[order(pred_NN_NE_cann)]
test[, plot_id := 1:.N]
test %>% sample_n(2000) %>% melt.data.table(measure.vars = paste0("pred_", c("GLM", "NN", "NN_NE", "NN_NE_cann"))) %>% 
  ggplot(aes(x = plot_id, y = value))+geom_line(aes(group = variable, colour = variable))+
  facet_wrap(~variable)+theme_pubr()

test[order(-abs(pred_NN_NE - pred_GLM))][1:10]
test[order(-abs(pred_NN_NE - pred_NN))][1:10]

test %>%  melt.data.table(measure.vars = c(paste0("pred_", c("GLM", "NN", "NN_NE", "NN_NE_cann")), "ClaimNb")) %>% 
  .[, .(freq = sum(value)/sum(Exposure)), by = .(variable, DrivAge)] %>% 
  ggplot(aes(x = DrivAge, y = freq))+geom_line(aes(group = variable, colour = variable))+
  facet_wrap(~variable)+theme_pubr()
