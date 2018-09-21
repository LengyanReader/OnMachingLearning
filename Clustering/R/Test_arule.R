# associated rules

# gorcery/merchant recommandation
library(arules)






devtools::install_github("rstudio/keras")




#-----------------------------------------

# advanced R 

library(data.table)

set.seed(1014)
df <- data.frame(replicate(6, sample(c(1:10, -99), 6, rep = TRUE)))
names(df) <- letters[1:6]
df<-data.table(df)

fix_missing <- function(x) {
  x[x == -99] <- NA
  x
}
df[] <- lapply(df, fix_missing)







df[1:5] <- lapply(df[1:5], fix_missing)






#--------------------------------------------

new_counter <- function() {
  i <- 0
  function() {
    i <<- i + 1
    i
  }
}


counter_one <- new_counter()
counter_two <- new_counter()

counter_one()







new_counter <- function() {
  i <- 0
  function() {
    i <- i + 1
    i
  }
}


counter_one <- new_counter()
counter_two <- new_counter()

counter_one()


#-----------------------------------

compute_mean <- list(
  base = function(x) mean(x),
  sum = function(x) sum(x) / length(x),
  manual = function(x) {
    total <- 0
    n <- length(x)
    for (i in seq_along(x)) {
      total <- total + x[i] / n
    }
    total
  }
)



x <- runif(1e7)
system.time(compute_mean$base(x))
system.time(compute_mean[[2]](x))

system.time(compute_mean[["manual"]](x))









