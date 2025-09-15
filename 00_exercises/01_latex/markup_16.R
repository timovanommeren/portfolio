#load package lattice
library(lattice)
library(xtable) # generate the LaTeX code for tables
#fix the random generator seed
set.seed(123)
#create data
data <- rnorm(1000)

plots <- list()

#plot histogram
plots[[1]] <- histogram(data)
#plot density 
plots[[2]] <-densityplot(data^12 / data^10, xlab = expression(data^12/data^10))
#plot stripplot
plots[[3]] <-stripplot(data^2, xlab = expression(data^2))
#plot boxplot
plots[[4]] <-bwplot(exp(data))

for (i in seq_along(plots)) {
  
  png(sprintf("myplot_%d.png", i), width = 800, height = 600, res = 150)
  
  print(plots[[i]])
  
  dev.off()
}

#matrix with all data used
data.all <- cbind(data = data, 
                  squared1 = data^12 / data^10,
                  squared2 = data^2,
                  exponent = exp(data))


write(print(xtable(data.all[1:9,]), type="latex"), file = "table.txt")
