## prep
# wd
cd("$(homedir())/Documents/GitHub Projects/ExpertElicitation/julia code/idyllic")
pwd()

# parallel
using Distributed
addprocs(3)

# library packages
@everywhere using Random, Distributions, DataFrames, LinearAlgebra, ProgressMeter, RCall, RData, PolyaGammaSamplers

# data
data_list = RData.load("data_list.rds")

## helper functions
@everywhere function sample_pg(n, b, eta; trunc = 1000)

  out = zeros(n)
  for i in 1:n
    if isinteger(b[i])
      s = PolyaGammaSamplers.PolyaGammaPSWSampler(Int(b[i]), eta[i])
      out[i] = rand(s, 1)[1]
    else
      tmp = 0
      for k in 1:trunc
        tmp += rand(Gamma(b[i], 1)) / ((k -.5)*(k-.5) + eta[i]*eta[i] / (4*pi*pi))
      end
      out[i] = (1 / (2*pi*pi)) * tmp
    end
  end

  return(out)
end
function mcmctoR(mcmc, filename)
  out = Dict()
  for i in 1:length(mcmc)
    out[string("chain", i)] = mcmc[i]
  end
  @rput out
  @rput filename
  R"saveRDS(out, file = filename)"
end
function export_to_R(dict, filename)
  @rput dict
  @rput filename
  R"saveRDS(dict, file = filename)"
end

@everywhere function fit_occ(seed, num_mcmc, thin, data_list; warmup = num_mcmc/2)

	# error checking
	if (num_mcmc % thin) != 0
		error("Error: Number of iterations not divisible by thinning interval.")
	end

	# set seed
	Random.seed!(seed)
	τ_a, τ_b = (1, 1)

	# housekeeping
	X = data_list["X"]
	y = data_list["y"]
	J = data_list["J"]
	C = data_list["C"]
 	μ = data_list["expert_means"]
	σ = data_list["expert_sds"]
	Σ = Diagonal(σ)
	Σinv = inv(Σ)
	Σinvμ = Σinv * μ
	n = size(X)[1]
	p = size(X)[2]
	K = size(μ, 1)

	# storage
	τ_mcmc = zeros(Int((num_mcmc - warmup)/thin), 1)
	β_mcmc = zeros(Int((num_mcmc - warmup)/thin), p)
	θ_mcmc = zeros(Int((num_mcmc - warmup)/thin), K)
	η_mcmc = zeros(Int((num_mcmc - warmup)/thin), n)
	z_mcmc = zeros(Int((num_mcmc - warmup)/thin), n)
	p_mcmc = zeros(Int((num_mcmc - warmup)/thin), 1)

	# intialize
	τ = first(rand(InverseGamma(τ_a, τ_b)))
	τ2 = dot(τ, τ)
	β = rand(MvNormal(zeros(p), 2 .* I(p)))
	θ = vec(rand(MvNormal(μ, Σ), 1))
	η = vec(rand(MvNormal(vec(C*θ), Diagonal(τ2 .* ones(n))), 1))
	linpred = X * β + η
	ψ = exp.(linpred) ./ (1 .+ exp.(linpred))
	z = fill(0, n)
	for i in 1:n
		z[i] = first(rand(Binomial(1, ψ[i]), 1))
	end
	detect_p = .5

	# convenience
	Xt = transpose(X)
	XtX = transpose(X) * X
	Ct = transpose(C)
	CtC = transpose(C) * C


	# sample
	progress_bar = Progress(num_mcmc, dt = 1, desc = string("Progress: "))
	for iter in 1:num_mcmc

		# sample z
		for i in 1:n
			if y[i] != 0
				z[i] = 1
			else
				tmp = ψ[i] * detect_p^J
				z[i] = first(rand(Binomial(1, tmp / (1 - ψ[i] + tmp))))
			end
		end

		# pg stuff
		κ = z .- .5
		ω = sample_pg(n, ones(n), X * β + η)
		Ω = Diagonal(ω)
		z_ = κ ./ ω

		# beta
		V_β = inv(Xt * Ω * X + Diagonal(.5 .* ones(p))); V_β = Symmetric(V_β)
		m_β = V_β * (Xt * Ω * (z_ - η))
		β = rand(MvNormal(m_β, V_β), 1)

		# eta
		V_η = inv(Ω + Diagonal(1/τ2 .* ones(n))); V_η = Symmetric(V_η)
		m_η = vec(V_η * (Ω * (z_ - X*β) + 1/τ2 .* C * θ))
		η = rand(MvNormal(m_η, V_η))

		# theta
		V_θ = inv(1/τ2 .* CtC + Σinv); V_θ = Symmetric(V_θ)
		m_θ = vec(V_θ * (1/τ2 .* Ct * η + Σinvμ))
		θ = rand(MvNormal(m_θ, V_θ), 1)

		# tau
		# SSE = first(transpose(η - C * θ) * (η - C * θ))
		# a = τ_a + n/2
		# b = τ_b + .5 * SSE
		# τ2 = rand(InverseGamma(a, b))
		# τ = sqrt(τ2)
		τ2 = 1
		τ = 1

		# p
		a = 1 + dot(z, y)
		b = 1 + dot(z, fill(J, n) - y)
		detect_p = first(rand(Beta(a, b)))

		# store, accounting for thinning
		if (iter > warmup) & ((iter-warmup) % thin == 0)
			τ_mcmc[Int((iter-warmup)/thin),1] = τ
			β_mcmc[Int((iter-warmup)/thin),1:p] = β[1:p]
			θ_mcmc[Int((iter-warmup)/thin),1:K] = θ[1:K]
			η_mcmc[Int((iter-warmup)/thin),1:n] = η
			z_mcmc[Int((iter-warmup)/thin),1:n] = z
			p_mcmc[Int((iter-warmup)/thin),1] = detect_p
		end

		# increment progress
		next!(progress_bar)
	end

	# names
	etanames = repeat(["f"], n)
	for et in 1:n
		etanames[et] = string("eta[", et, "]")
	end

	betanames = repeat(["f"], p)
	for ndx in 1:p
		betanames[ndx] = string("beta[", ndx, "]")
	end

	thetanames = repeat(["f"], K)
	for ndx in 1:K
		thetanames[ndx] = string("theta[", ndx, "]")
	end

	znames = repeat(["f"], n)
	for ndx in 1:n
		znames[ndx] = string("z[", ndx, "]")
	end

	names = ["tau"; betanames; thetanames; etanames; znames; "p"]
	out = Dict()
	out["samples"] = hcat(τ_mcmc, β_mcmc, θ_mcmc, η_mcmc, z_mcmc, p_mcmc)
	out["names"] = names

	return(out)
end

# fit model
@everywhere curry(f, xs...) = y -> f(y, xs...)
fit_parallel = pmap(curry(fit_occ, 50000, 5, data_list), [1 2 3])
mcmctoR(fit_parallel, "expert_hier_idyllic.rds")
