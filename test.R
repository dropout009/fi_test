# load package ------------------------------------------------------------


library(tidyverse)
library(tidymodels)

library(scales)
library(furrr)
library(glue)



library(ggExtra)
library(ggforce)


library(latex2exp)
library(tidylog)
library(copula)


# theme -------------------------------------------------------------------



modern = c("#323a45","#3f6184","#778899","#f6f7f9","#5faeb6")


theme_modern = function() {
  theme_minimal(base_size = 12, base_family = "Open Sans") %+replace%
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(color = modern[3], size = 0.1),
          axis.title = element_text(size = 15, color = modern[1]),
          axis.title.x = element_text(margin = margin(10, 0, 0, 0), hjust = 1),
          axis.title.y = element_text(margin = margin(0, 10, 0, 0), angle = 90, hjust = 1),
          axis.text = element_text(size = 12, color = modern[1]),
          strip.text = element_text(size = 15, color = modern[1], margin = margin(5, 5, 5, 5)),
          plot.title = element_text(size = 18, color = modern[1], margin = margin(0, 0, 18, 0)))
}


#\begin{aligned} y_{i}=& x_{i 1}+x_{i 2}+x_{i 3}+x_{i 4}+x_{i 5}+0 x_{i 6} \\ &+0.5 x_{i 7}+0.8 x_{i 8}+1.2 x_{i 9}+1.5 x_{i 10}+\epsilon_{i} \end{aligned}

R = 10
N = 1000
D = 10

beta = c(1, 1, 1, 1, 1, 0, 0.5, 0.8, 1.2, 1.5)

df_true = tibble(beta = beta) %>% 
  mutate(feature = row_number(),
         n = rank(desc(beta)))

df_result = tibble()
for (rho in c(0, 0.9)) {
  
  copula = normalCopula(param=rho, dim = 2) %>% 
    mvdc(margins = c("unif", "unif"), 
         list(list(min=0, max=1),
              list(min=0, max=1)))
  
  for (r in 1:R) {
    
    x1_2 = rMvdc(N, copula) %>% 
      as_tibble()
    
    x3_10 = runif(N * 8, min = 0, max=1) %>% 
      matrix(nrow = N, ncol = 8) %>% 
      as_tibble()
    
    x = x1_2 %>% 
      bind_cols(x3_10) 
    
    colnames(x) = c("x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10")
    
    df = x %>% 
      mutate(y = c(as.matrix(x) %*% beta + rnorm(N, mean = 0, sd = 0.1)))
    
    
    split = initial_split(df, prop = 0.5)
    
    df_train = training(split)
    df_test = testing(split)
    
    rf = rand_forest(trees = 1000) %>% 
      set_engine(engine = "ranger", 
                 num.threads = parallel::detectCores(), 
                 seed = 42) %>% 
      fit(y ~ ., data = df_train)
    
    ols = linear_reg() %>% 
      set_engine("lm") %>% 
      fit(y ~ ., data = df_train)
    
    for (m in c("ols", "rf")) {
      df_pred = df_test %>% select(y)
      for (d in 1:D) {
        df_temp = df_test
        df_temp[d] = sample(df_temp[d] %>% pull())
        
        if (m == "ols") {
          df_pred = df_pred %>% 
            bind_cols(predict(ols, df_temp))
          
        } else {
          df_pred = df_pred %>% 
            bind_cols(predict(rf, df_temp))
          
        }
        
      }  
      colnames(df_pred) = c("y", "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10")
      tmp_result = df_pred %>% 
        gather(feature, value, -y) %>% 
        group_by(feature) %>% 
        rmse(y, value) %>% 
        mutate(n = rank(desc(.estimate)),
               model = m,
               round = r,
               rho = rho)
      
      df_result = df_result %>% 
        bind_rows(tmp_result)
    }  
  }
}


df_result %>%
  filter(model == "rf") %>% 
  mutate(feature = str_remove(feature, "x") %>% as.integer()) %>% 
  group_by(feature, model, rho) %>% 
  summarise(n = mean(n)) %>% 
  ggplot(aes(feature, n)) +
  geom_line(data = df_true, color = modern[3], size = 1) + 
  geom_line(color = modern[2], size = 1) +
  geom_point(color = modern[2], size = 3) +
  facet_wrap(~rho, labeller = "label_both") +
  scale_x_continuous(breaks =  seq(1, 10, 1)) + 
  scale_y_reverse(breaks = seq(1, 10, 2)) +
  labs(x = "Feature No.", y = "Importance Rank") + 
  theme_modern()




df_result2 = tibble()
for (rho in c(0, 0.9)) {
  
  copula = normalCopula(param=rho, dim = 2) %>% 
    mvdc(margins = c("unif", "unif"), 
         list(list(min=0, max=1),
              list(min=0, max=1)))
  
  for (r in 1:R) {
    
    x1_2 = rMvdc(N, copula) %>% 
      as_tibble()
    
    x3_10 = runif(N * 8, min = 0, max=1) %>% 
      matrix(nrow = N, ncol = 8) %>% 
      as_tibble()
    
    x = x1_2 %>% 
      bind_cols(x3_10) 
    
    colnames(x) = c("x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10")
    
    df = x %>% 
      mutate(y = c(as.matrix(x) %*% beta + rnorm(N, mean = 0, sd = 0.1)))
    
    
    split = initial_split(df, prop = 0.5)
    
    df_train = training(split)
    df_test = testing(split)
  
    for (m in c("ols", "rf")) {
      df_pred = df_test %>% select(y)
      for (d in 1:D) {
        
        df_train_temp = df_train[-d]
        #df_train_temp[d] = sample(df_train_temp[d] %>% pull())
        
        df_test_temp = df_test[-d]
        #df_test_temp[d] = sample(df_test_temp[d] %>% pull())
        
        
        rf = rand_forest(trees = 1000) %>% 
          set_engine(engine = "ranger", 
                     num.threads = parallel::detectCores(), 
                     seed = 42) %>% 
          fit(y ~ ., data = df_train_temp)
        
        ols = linear_reg() %>% 
          set_engine("lm") %>% 
          fit(y ~ ., data = df_train_temp)
        
        
        if (m == "ols") {
          df_pred = df_pred %>% 
            bind_cols(predict(ols, df_test_temp))
          
        } else {
          df_pred = df_pred %>% 
            bind_cols(predict(rf, df_test_temp))
          
        }
        
      }  
      colnames(df_pred) = c("y", "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10")
      tmp_result = df_pred %>% 
        gather(feature, value, -y) %>% 
        group_by(feature) %>% 
        rmse(y, value) %>% 
        mutate(n = rank(desc(.estimate)),
               model = m,
               round = r,
               rho = rho)
      
      df_result2 = df_result2 %>% 
        bind_rows(tmp_result)
    }  
  }
}


df_result2 %>%
  filter(model == "rf") %>% 
  mutate(feature = str_remove(feature, "x") %>% as.integer()) %>% 
  group_by(feature, model, rho) %>% 
  summarise(n = mean(n)) %>% 
  ggplot(aes(feature, n)) +
  geom_line(data = df_true, color = modern[3], size = 1) + 
  geom_line(color = modern[2], size = 1) +
  geom_point(color = modern[2], size = 3) +
  facet_wrap(~rho, labeller = "label_both") +
  scale_x_continuous(breaks =  seq(1, 10, 2)) + 
  scale_y_reverse(breaks = seq(1, 10, 2)) +
  labs(x = "Feature No.", y = "Importance Rank") + 
  theme_modern()

ggsave("figure/drop_01.png", width = 6, height = 4, dpi = "retina")



df1 = tibble(x = 1:5, y = c(11, 11, 12, 13, 14))
df1 %>% 
  filter((x ==2 & y <= 12 | y >= 14)) 







