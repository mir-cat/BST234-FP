#####################
### BST 234       ###
### Final project ### 
#####################

setwd("/home/kc171/R")
#setwd("C:/Users/Katharine/Documents/Harvard/Spring 2017/BIOSTAT234/Final Project")

proj.data <- read.csv("data.csv")

set.seed(20161202)

S <- 1000

#######################
# Logistic Score ----------------------------------------------------------
#######################


# fit null model
logistic.null.fit <- glm(Y ~ Age + Female + Smoker + PC1 + PC2
                           ,family=binomial(link='logit')
                           ,data=proj.data)

# calculate observed score statistic
p.hat.null <- logistic.null.fit$fitted.values
res.null <- as.vector(proj.data$Y - p.hat.null)
X.1 <- as.vector(proj.data$X_1)
X.1sq <- as.vector(proj.data$X_1*proj.data$X_1)
Var.null <- p.hat.null*(1-p.hat.null)
  
observed.score.num <- as.numeric((t(res.null)%*%X.1)^2)
observed.score.denom <- as.numeric(t(X.1sq)%*%Var.null)
observed.score <- observed.score.num/observed.score.denom

# Start the clock!
ptm1 <- proc.time()

# conduct permutation test
permuted.score <- rep(NA,S)
for (s in 1:S){
    # permute the exposure (pick # of 1,2s random spots to put the 
    # rare allelles to shuffle the exposure)
    one.spots <- sample(1:n
                        , size=num.allele1
                        , replace = FALSE, prob = NULL)
    two.spots <- sample((1:n)[!(1:n %in% one.spots)]
                        , size=num.allele2
                        , replace = FALSE, prob = NULL)
    X.1.perm <- rep(0,n)
    X.1.perm[one.spots] <- 1
    X.1.perm[two.spots] <- 2
    
    # calculate the permuted score stat
    X.1.perm.sq <- as.vector(X.1.perm*X.1.perm)
    
    permuted.score.num <- as.numeric((t(res.null)%*%X.1.perm)^2)
    permuted.score.denom <- as.numeric(t(X.1.perm.sq)%*%Var.null)
    permuted.score[s] <- permuted.score.num/permuted.score.denom  
}


# Stop the clock
ptm2 <- proc.time() 

ptm2 - ptm1


# sum the number of permuted estimates that were more extreme than the observed estimate
# (note that the score statistic will be positive given numerator is squared and denominator
# involves summing a squared value times positive values)
p.val <- (length(which(permuted.score > observed.score))/(S+1))

hist(permuted.score)
p.val

#######################
# 1. Other ways to permute ----------------------------------------------------------
#######################

# Start the clock!
ptm3 <- proc.time()

# conduct permutation test by subsetting only on where permuted 1s and 2s are
permuted.score2 <- rep(NA,S)
for (s in 1:S){
  # permute the exposure (pick # of 1,2s random spots to put the 
  # rare allelles to shuffle the exposure)
  one.spots <- sample(1:n
                      , size=num.allele1
                      , replace = FALSE, prob = NULL)
  two.spots <- sample((1:n)[!(1:n %in% one.spots)]
                      , size=num.allele2
                      , replace = FALSE, prob = NULL)

  permuted.score.num <- (sum(res.null[one.spots]*1) 
                         + sum(res.null[two.spots]*2))^2
                                                         
  permuted.score.denom <- (sum(Var.null[one.spots]*1) +
                             sum(Var.null[two.spots]*4))
  permuted.score2[s] <- permuted.score.num/permuted.score.denom  
}


# Stop the clock
ptm4<- proc.time() 

ptm4-ptm3

hist(permuted.score)
hist(permuted.score2)

# sum the number of permuted estimates that were more extreme than the observed estimate
# (note that the score statistic will be positive given numerator is squared and denominator
# involves summing a squared value times positive values)
p.val2 <- (length(which(permuted.score > observed.score))/(S+1))

p.val2

#######################
# 2.compare values ----------------------------------------------------------
#######################

# conduct permutation test
permuted.score <- rep(NA,S)
permuted.score2 <- rep(NA,S)
for (s in 1:S){
  # permute the exposure (pick # of 1,2s random spots to put the 
  # rare allelles to shuffle the exposure)
  one.spots <- sample(1:n
                      , size=num.allele1
                      , replace = FALSE, prob = NULL)
  two.spots <- sample((1:n)[!(1:n %in% one.spots)]
                      , size=num.allele2
                      , replace = FALSE, prob = NULL)
  X.1.perm <- rep(0,n)
  X.1.perm[one.spots] <- 1
  X.1.perm[two.spots] <- 2
  
  # calculate the permuted score stat
  X.1.perm.sq <- as.vector(X.1.perm*X.1.perm)
  
  permuted.score.num <- as.numeric((t(res.null)%*%X.1.perm)^2)
  permuted.score.denom <- as.numeric(t(X.1.perm.sq)%*%Var.null)
  permuted.score[s] <- permuted.score.num/permuted.score.denom 
  
  permuted.score.num2 <- (sum(res.null[one.spots]*1) 
                         + sum(res.null[two.spots]*2))^2
  
  permuted.score.denom2 <- (sum(Var.null[one.spots]*1) +
                             sum(Var.null[two.spots]*4))
  permuted.score2[s] <- permuted.score.num2/permuted.score.denom2 
}


# compare values; yupp, the same
head(permuted.score)
head(permuted.score2)
tail(permuted.score)
tail(permuted.score2)
