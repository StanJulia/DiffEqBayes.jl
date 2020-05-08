using CmdStan, DiffEqBayes, OrdinaryDiffEq, ParameterizedFunctions,
      RecursiveArrayTools, Distributions, Random, Test

# Uncomment for local testing only, make sure MCMCChains is available
using MCMCChains

Random.seed!(123)

println("One parameter case")
f1 = @ode_def begin
  dx = a*x - x*y
  dy = -3y + x*y
end a
u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5]
prob1 = ODEProblem(f1,u0,tspan,p)
sol = solve(prob1,Tsit5())
t = collect(range(1,stop=10,length=10))
randomized = VectorOfArray([(sol(t[i]) + .5randn(2)) for i in 1:length(t)])
data = convert(Array,randomized)
priors = [truncated(Normal(0.7,1),0.1,2)]

bayesian_result = stan_inference(prob1,t,data,priors;nchains=4, num_samples=2000,
  num_warmup=1000,likelihood=Normal)

sdf  = CmdStan.read_summary(bayesian_result.model)
@test sdf[sdf.parameters .== :theta1, :mean][1] ≈ 1.5 atol=3e-1

# Uncomment for local chain inspection
chn = CmdStan.convert_a3d(bayesian_result.chains, bayesian_result.cnames, Val(:mcmcchains))
plot(chn)
!isdir("tmp") && mkdir("tmp")
savefig("$(@__DIR__)/tmp/one_parameter_case.png")

println("Four parameter case")
f1 = @ode_def begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a b c d
u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob1 = ODEProblem(f1,u0,tspan,p)
sol = solve(prob1,Tsit5())
t = collect(range(1,stop=10,length=10))
randomized = VectorOfArray([(sol(t[i]) + .5randn(2)) for i in 1:length(t)])
data = convert(Array,randomized)
priors = [truncated(Normal(1.0,1),0.1,2),truncated(Normal(1.5,0.5),0.1,1.5),
          truncated(Normal(2.0,1),0.1,4),truncated(Normal(1.3,0.5),0.1,2)]

bayesian_result = stan_inference(prob1,t,data,priors;num_samples=2000, nchains=4,
  num_warmup=1000,vars =(DiffEqBayes.StanODEData(),InverseGamma(4,1)))
sdf  = CmdStan.read_summary(bayesian_result.model)
@test sdf[sdf.parameters .== :theta1, :mean][1] ≈ 1.5 atol=3e-1
@test sdf[sdf.parameters .== :theta2, :mean][1] ≈ 1.0 atol=3e-1
@test sdf[sdf.parameters .== :theta3, :mean][1] ≈ 3.0 atol=3e-1
@test sdf[sdf.parameters .== :theta4, :mean][1] ≈ 1.0 atol=3e-1

# Uncomment for local chain inspection
chn = CmdStan.convert_a3d(bayesian_result.chains, bayesian_result.cnames, Val(:mcmcchains))
plot(chn)
savefig("$(@__DIR__)/tmp/four_parameter_case.png")


