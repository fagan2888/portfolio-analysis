library(quantmod)
library(sde)

set.seed(100)
syms=Cl(getSymbols("AAPL",from="2014-01-01",auto.assign=F))
ret=ROC(syms)

mu=mean(na.omit(ret))
sigma=sd(na.omit(ret))
print(mu)
print(sigma)

days = 255
P0 = coredata(tail(syms,n=1))

T = 1
nt=50; # trajectories
dt=T/days; t=seq(0,T,by=dt)
X=matrix(rep(0,length(t)*nt), nrow=nt)
for (i in 1:nt) {X[i,] = GBM(x=P0,r=mu,sigma=sigma,T=T,N=days)}

##Plot
ymax=max(X); ymin=min(X) #bounds for simulated prices
plot(t,X[1,],t='l',ylim=c(ymin, ymax), col=1,
    ylab="Price P(t)",xlab="time t")
for(i in 2:nt){lines(t,X[i,], t='l',ylim=c(ymin, ymax),col=i)}
