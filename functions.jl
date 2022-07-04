
function get_beta(nu)
	get_intensity(beta) = 1 / (n_prices * n_states) * beta / (1 - beta^(n_agents+1))
	beta = 0.9
	for prec in 5:16
		while nu - get_intensity(beta) > 10.0^-(prec+1)
	 		beta += 10.0^-prec
	 	end
	 	beta -= 10.0^-prec
	end
	return beta
end

function gen_equilibrium_prices()
	# compute nash and cooperation prices for symmetric fims (having the same a and c)
	nash_price = Array{Float32,2}(undef,n_agents,n_markets)
	coop_price = Array{Float32,2}(undef,n_agents,n_markets)	
	# nash price is a fixed point of the equation p = c + \mu / (1 - (n + exp[(a0 - a + p)/mu]))^(-1) see Anderson and De Palma (1992)   
	f(p,a,c,m,mu) = c + mu/(1 - (n_agents + exp((a0[m] - a + p) / mu))^(-1))  									# find fixed point of p = f(p)
    # coop price maximizes the firms' joint profits (here I assume symmetry, and solve single-agent problem)
    pf(p,a,c,m,mu) = - (p - c) * exp((a - p) / mu) / (n_agents * exp((a - p) / mu) + exp(a0[m] / mu))			# pf(p) = - \pi(p) thus argmax(\pi) = -argmin(\pi)
	dq(p,a,c,m,mu) = (pf(p + 1e-8,a,c,m,mu) - pf(p,a,c,m,mu)) / 1e-8 											# difference quotient 
	mu_ = ifelse(mu == 0, -1e-16, mu)																			# look for left limit if mu=0
	for m in 1:n_markets
		np = Float64.(a)
		cp = Float64.(a)
		# compute fixed point of p = f(p)  
	 	while any(abs.(np - f.(np,a,c,m,mu_)) .> 1e-8)
	 		np = f.(np,a,c,m,mu_)
	 	end
	 	# compute first order condition d\pi/dp = 0 
		while any(abs.(dq.(cp,a,c,m,mu_)) .> 1e-8) 
			cp -= 1.0e-5 * dq.(cp,a,c,m,mu_)
		end
		nash_price[:,m] = Float32.(round.(np, digits=7))
		coop_price[:,m] = Float32.(round.(cp, digits=7))
	end
	return nash_price, coop_price
end


function gen_prices()
	# compute discrete (linearly spaced) set of prices
	# the nash price is the (p_nash)-th price, the cooperation price is the (p_coop)-th price
	prices = Array{Float32,2}(undef,n_prices,n_agents);
	for i in 1:n_agents
		offset = (coop_price[i,1] - nash_price[i,n_markets])/(n_prices-2*(p_nash[i]-1)-1)
 		lower_bound = nash_price[i,n_markets] - (p_nash[i]-1) * offset
 		upper_bound = coop_price[i,1] + (p_nash[i]-1) * offset
		#offset = (coop_price[i,1] - nash_price[i,n_markets]) / (n_prices - p_nash[i])
		#lower_bound = nash_price[i,n_markets] - (p_nash[i]-1) * offset
		#upper_bound = coop_price[i,1]
		prices[:,i] = collect(lower_bound:offset:upper_bound+0.001f0) #collect(range(lower_bound,upper_bound,n_prices))
	end
	return prices
end


function profit_function(p,m)
	# compute profits for given set of prices and market
	mu_ = mu	
	while true																								# do while no limit equals NaN nor Inf (dynamic precision)
		profit = (p .- c) .* exp.((a .- p) / mu_) ./ (sum(exp.((a .- p) / mu_)) .+ exp.(a0[m] / mu_))
		mu_ += 1e-6																							# decrease precision
		any(isnan.(profit)) || any(isinf.(profit)) || break													# when mu is close to 0 numerical limits are required
	end	
	if mu == 0 && isequal(p, coop_price[:,m])																# cooperation profits have sketchy limit for mu = 0
		mu_ = 1e-17					
		p_	= p .- 1e-15																		
		profit = (p_ .- c) .* exp.((a .- p_) / mu_) ./ (sum(exp.((a .- p_) / mu_)) .+ exp.(a0[m] / mu_))
	end
	return Float32.(round.(profit, digits=7))
end


function gen_equilibrium_profits()
	# compute nash and cooperation profits for each market
	# profits = (p - c) q
	nash_profit = Array{Float32,2}(undef,n_agents,n_markets)
	coop_profit = Array{Float32,2}(undef,n_agents,n_markets)																	
	for m in 1:n_markets
		nash_profit[:,m] = profit_function(nash_price[:,m],m)
		coop_profit[:,m] = profit_function(coop_price[:,m],m)
	end
	return nash_profit, coop_profit
end


function gen_payoff_matrix()
	# compute payoff matrices, one for each market
	dims = ntuple(d -> n_prices, Val(n_agents))
	payoffs = Array{NTuple{n_agents,Float32}, n_agents}(undef, dims)
	payoffs_market = Array{typeof(payoffs)}(undef, n_markets)
	expected_payoffs =  Array{NTuple{n_agents,Float32}}(undef, dims)
	for m in 1:n_markets
		for p_ in CartesianIndices(payoffs)
			p = CartesianIndex.([(Tuple(p_)[i],i) for i = 1:n_agents])
			payoffs[p_] = Tuple(profit_function(prices[p],m))
		end
		payoffs_market[m] = copy(payoffs)
	end
	expected_payoffs = Tuple.(mean(collect.(payoffs_market[m]) for m in 1:n_markets))
	return payoffs_market, expected_payoffs
end


function gen_one_memory_states()
	memory_length == 0 && return undef				# early return if memory_length = 0
	# generate coarse partition of prices
	coarse_prices = Vector{Vector{Int8}}()
	start_ = 1
	for length in c_part
	  end_ = start_ + length - 1
	  push!(coarse_prices, price_numbers[start_:end_])
	  start_ += length
	end

	# generate one memory (possibly coarse) states number (or strategy profile)
	# one_memory_state asssign a unique number to any combination of prices given agent's i coarse memory
	dims = Tuple(pushfirst!([length(c_part) for i in 1:n_agents-1], n_prices))
	temp = Array{Int32, n_agents}(undef, dims)
	for i in eachindex(temp)
	  temp[i] = i
	end

	dims = ntuple(d -> n_prices, Val(n_agents))
	one_memory_state = Array{Array{Int32},n_agents}(undef, dims)
	subjective_state = Array{Int32,1}(undef, n_agents)
	for i in keys(one_memory_state)
		p = collect(Tuple(i))
		for i_agent in 1:n_agents
			p_ = copy(p)
			for j in 2:n_agents
				p_[j] = get_coarse_price(p_[j],coarse_prices)
			end         
			subjective_state[i_agent] = temp[p_...]
			p = circshift(p,-1)
		end
		one_memory_state[i] = copy(subjective_state)
	end
	return one_memory_state
end


function get_coarse_price(p, coarse_prices::Array{Array{Int8,1},1})
   for p_c in eachindex(coarse_prices)
      for p_ in eachindex(coarse_prices[p_c])
         if p == coarse_prices[p_c][p_]
            return p_c
         end
      end
   end
end


function gen_state_numbers()
	# generate state number
	# if memory_length = 0, there is only one state
	# if memory_length = 1, then state_number = one_memory_state
	# if memory_length > 1, then state_number assigns a unique number to any combination of states in memory
	dims = ntuple(d -> Int64(one_memory_n_states), Val(memory_length))
	state_number = Array{Int32,memory_length}(undef, dims)
	for i in eachindex(state_number)
		state_number[i] = i
	end
	return state_number
end


function get_tuple(idx::AbstractArray, n::Int64)
	# quick non-allocating array to tuple index conversion
	return ntuple(i -> idx[i], n)
end


function get_profits(m::Int8, p::Array{Int8})
	# get agents' payoff given market m and prices p
	return payoffs[m][get_tuple(p,n_agents)...]
end


function get_expected_profits(p::Array{Int8})
	# get agents' expected payoff given prices p
	return collect(expected_payoffs[get_tuple(p,n_agents)...])
end


function get_one_memory_state(p::Array{Int8})
	# get state number from price tuple
	return one_memory_state[get_tuple(p,n_agents)...]
end


function get_state_number(memory::Array{Int32,2})
	# get state number from memory
	if memory_length <= 1
		return memory[:]					# state = memory
	elseif memory_length > 1
		state = Array{Int32,1}(undef, n_agents)
		for i in 1:n_agents
			state[i] = state_number[get_tuple(memory[i,:],memory_length)...] 
		end
		return state
	end
end


function argmax_(A::AbstractArray, maximizer::Array{Int8,1})
	# efficiently find set of maximizers of array A
	maximum = -Inf32
	n = 0
	for i in keys(A)
		A_i = A[i]
		A_i < maximum && continue
		if A_i > maximum
			maximum = A_i
			n = 1
			maximizer[n] = i
		else
			maximizer[n+=1] = i
		end
	end
	return view(maximizer,1:n)
end
