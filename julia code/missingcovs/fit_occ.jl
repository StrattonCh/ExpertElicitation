## prep
# wd
cd("$(homedir())/Documents/GitHub Projects/ExpertElicitation/julia code/missingcovs")
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

## model0
@everywhere function fit_occ(seed, num_mcmc, thin, data_list; warmup = num_mcmc/2)

	# error checking
	if (num_mcmc % thin) != 0
		error("Error: Number of iterations not divisible by thinning interval.")
	end

	# set seed
	Random.seed!(seed)

	# housekeeping
	X = data_list["X"]
	y = data_list["y"]
	J = data_list["J"]
	n = size(X)[1]
	p = size(X)[2]

	# storage
	β_mcmc = zeros(Int((num_mcmc - warmup)/thin), p)
	ψ_mcmc = zeros(Int((num_mcmc - warmup)/thin), n)
	p_mcmc = zeros(Int((num_mcmc - warmup)/thin), 1)

	# intialize
	β = rand(MvNormal(zeros(p), 2 .* I(p)))
	linpred = X * β
	ψ = exp.(linpred) ./ (1 .+ exp.(linpred))
	z = fill(0, n)
	for i in 1:n
		z[i] = first(rand(Binomial(1, ψ[i]), 1))
	end
	detect_p = .5

	# convenience
	Xt = transpose(X)
	XtX = transpose(X) * X

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
		ω = sample_pg(n, ones(n), X * β)
		Ω = Diagonal(ω)
		z_ = κ ./ ω

		# beta
		V_β = inv(Xt * Ω * X + Diagonal(.5 .* ones(p))); V_β = Symmetric(V_β)
		m_β = vec(V_β * (Xt * κ))
		β = rand(MvNormal(m_β, V_β), 1)
		linpred = X * β
		ψ = exp.(linpred) ./ (1 .+ exp.(linpred))

		# p
		a = 1 + dot(z, y)
		b = 1 + dot(z, fill(J, n) - y)
		detect_p = first(rand(Beta(a, b)))

		# store, accounting for thinning
		if (iter > warmup) & ((iter-warmup) % thin == 0)
			β_mcmc[Int((iter-warmup)/thin),1:p] = β[1:p]
			ψ_mcmc[Int((iter-warmup)/thin),1:n] = ψ
			p_mcmc[Int((iter-warmup)/thin),1] = detect_p
		end

		# increment progress
		next!(progress_bar)
	end

	# names
	betanames = repeat(["f"], p)
	for ndx in 1:p
		betanames[ndx] = string("beta[", ndx, "]")
	end

	ψnames = repeat(["f"], n)
	for ndx in 1:n
		ψnames[ndx] = string("psi[", ndx, "]")
	end

	names = ["p"; betanames; ψnames]
	out = Dict()
	out["samples"] = hcat(p_mcmc, β_mcmc, ψ_mcmc)
	out["names"] = names

	return(out)
end

# fit model
@everywhere curry(f, xs...) = y -> f(y, xs...)
fit_parallel = pmap(curry(fit_occ, 10000, 2, data_list), [1 2 3])
mcmctoR(fit_parallel, "model0.rds")


## model1
@everywhere function fit_occ_hier(seed, num_mcmc, thin, data_list; warmup = num_mcmc/2)

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
 	m = data_list["expert_means"]
	s = data_list["expert_sds"]
	Σ = Diagonal(s .* s)
	Σinv = inv(Σ)
	Σinvm = Σinv * m
	n = size(X)[1]
	p = size(X)[2]
	K = size(m, 1)

	# storage
	τ_mcmc = zeros(Int((num_mcmc - warmup)/thin), 1)
	β_mcmc = zeros(Int((num_mcmc - warmup)/thin), p)
	θ_mcmc = zeros(Int((num_mcmc - warmup)/thin), K)
	μ_mcmc = zeros(Int((num_mcmc - warmup)/thin), K)
	ψ_mcmc = zeros(Int((num_mcmc - warmup)/thin), n)
	p_mcmc = zeros(Int((num_mcmc - warmup)/thin), 1)

	# intialize
	τ = first(rand(InverseGamma(τ_a, τ_b)))
	τ2 = dot(τ, τ)
	μ = vec(rand(MvNormal(m, Σ)))
	β = rand(MvNormal(zeros(p), 2 .* I(p)))
	θ = vec(rand(MvNormal(μ, τ2 .* I(K)), 1))
	linpred = X * β + C * θ
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
		ω = sample_pg(n, ones(n), X * β + C * θ)
		Ω = Diagonal(ω)
		z_ = κ ./ ω

		# beta
		V_β = inv(Xt * Ω * X + Diagonal(.5 .* ones(p))); V_β = Symmetric(V_β)
		m_β = vec(V_β * (Xt * Ω * (z_ - C * θ)))
		β = rand(MvNormal(m_β, V_β), 1)

		# theta
		V_θ = inv(Ct * Ω * C + (1/τ2) .* I(K)); V_θ = Symmetric(V_θ)
		m_θ = vec(V_θ * (Ct * Ω * (z_ - X * β) + (1/τ2) .* μ))
		θ = rand(MvNormal(m_θ, V_θ), 1)
		linpred = X * β + C * θ
		ψ = exp.(linpred) ./ (1 .+ exp.(linpred))

		# mu
		V_μ = inv(1/τ2 .* I(K) + Σinv); V_μ = Symmetric(V_μ)
		m_μ = vec(V_μ * (1/τ2 .* θ + Σinvm))
		μ = rand(MvNormal(m_μ, V_μ))

		# tau
		SSE = first(transpose(θ - μ) * (θ - μ))
		a = τ_a + K/2
		b = τ_b + .5 * SSE
		τ2 = rand(InverseGamma(a, b))
		τ = sqrt(τ2)

		# p
		a = 1 + dot(z, y)
		b = 1 + dot(z, fill(J, n) - y)
		detect_p = first(rand(Beta(a, b)))

		# store, accounting for thinning
		if (iter > warmup) & ((iter-warmup) % thin == 0)
			τ_mcmc[Int((iter-warmup)/thin),1] = τ
			β_mcmc[Int((iter-warmup)/thin),1:p] = β[1:p]
			θ_mcmc[Int((iter-warmup)/thin),1:K] = θ[1:K]
			μ_mcmc[Int((iter-warmup)/thin),1:K] = μ[1:K]
			ψ_mcmc[Int((iter-warmup)/thin),1:n] = ψ
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

	ψnames = repeat(["f"], n)
	for ndx in 1:n
		ψnames[ndx] = string("psi[", ndx, "]")
	end

	names = ["p";"tau"; betanames; thetanames; ψnames]
	out = Dict()
	out["samples"] = hcat(p_mcmc, τ_mcmc, β_mcmc, θ_mcmc, ψ_mcmc)
	out["names"] = names

	return(out)
end

# fit model
@everywhere curry(f, xs...) = y -> f(y, xs...)
fit_parallel = pmap(curry(fit_occ_hier, 10000, 2, data_list), [1 2 3])
mcmctoR(fit_parallel, "expert_hier_idyllic.rds")
