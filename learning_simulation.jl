
function learning_simulation(z::Int64; seed = Random.seed!(z))
	# pre-allocate arrays, increases performance and reduces allocations
	alloc_p = Array{Int8,1}(undef, n_agents)
	alloc_max_idx = Array{Int8,1}(undef, n_prices)
	# initialize variables
	last_epoch = max_epochs
	same_greedy_policy = 0
	epoch_profits = Array{Float32,2}(undef, max_epochs, n_agents)
	# begin simulation
	Q = init_q_matrix()
	V, greedy_policy = init_greedy_policy(Q, alloc_max_idx)
	memory = init_memory()
	for epoch in 1:max_epochs
		new_greedy_policy, epoch_profits[epoch,:] = epoch_play(Q, V, copy(greedy_policy), epsilon[epoch], memory, alloc_p, alloc_max_idx)
		if isequal(greedy_policy, new_greedy_policy)
			same_greedy_policy += 1
			if same_greedy_policy == convergence_target
				last_epoch = epoch
				break
			end
		else
			same_greedy_policy = 0
			greedy_policy = copy(new_greedy_policy)
		end
	end
	return memory, last_epoch, greedy_policy, Q, epoch_profits
end


function epoch_play(Q::Array{Float32,3}, V::Array{Float32,2}, greedy_policy::Array{Int32,2}, epsilon::Array{Array{Float64,1},1}, memory::Array{Int32,2}, p::Array{Int8,1}, alloc_max_idx::Array{Int8,1})
	epoch_profits = zeros(Float32, n_agents)
	state = get_state_number(memory)
	market = Int8(rand(1:n_markets))
	for episode in 1:n_episodes
		p = get_price(state, greedy_policy, epsilon[episode], p = p)
		profits = get_profits(market, p)
		next_state = get_next_state(memory, p)
		Q = update_q(Q, V, state, p, profits, next_state, greedy_policy, alloc_max_idx)
		state = next_state
		epoch_profits .+= profits
	end
	return greedy_policy, epoch_profits
end


function update_q(Q::Array{Float32,3}, V::Array{Float32,2}, state::Array{Int32,1}, price::Array{Int8,1}, profits::Tuple, next_state::Array{Int32,1}, greedy_policy::Array{Int32,2}, alloc_max_idx::Array{Int8,1})
	# Q-learning update:  Q(s,a) <- (1 - alpha) * Q(s,a) + alpha * (R + gamma * V(s')) 
	#                     where V(s') = max_{a in A} Q(s',a) is the continuation value (value function)
	for i in 1:n_agents
		s, p = state[i], price[i]
		Q[s,p,i] = (1 - alpha[i]) * Q[s,p,i] + alpha[i] * (profits[i] + delta[i] * V[next_state[i],i])  # update Q
		if Q[s,p,i] <= V[s,i] && greedy_policy[s,i] == p
			greedy_policy[s,i] = rand(argmax_(view(Q,s,:,i),alloc_max_idx))						# current p[i] may not be best price anymore
		elseif Q[s,p,i] >= V[s,i] && greedy_policy[s,i] != p
			greedy_policy[s,i] = p															# current p[i] is best price in state s
		end
		V[s,i] = Q[s,greedy_policy[s,i],i]
	end
	return Q 			# Q is updated by reference, but returning its value increses readibility
end


function get_price(state::Array{Int32,1}, greedy_policy::Array{Int32,2}, epsilon::Array{Float64}; p = Array{Int8,1}(undef,n_agents))
	# get price given current greedy_policy under e-greedy policy
	for i in 1:n_agents
		p[i] = ifelse(rand() <= epsilon[i], rand(price_numbers), greedy_policy[state[i],i])
	end
	return p			# p is updated by reference, but returning its value increses readibility
end


function get_next_state(memory::Array{Int32,2}, p::Array{Int8,1})
	# update memory (by reference) and get next state number
	if memory_length == 0							# no need to update memory (always [1])
		return memory[:]							# state = memory[:] = (1,1,..)
	elseif memory_length == 1
		state = get_one_memory_state(p)				# state = memory
		memory[:] = state							# this is necessary to return last_memory 
	elseif memory_length > 1 										
		memory[:,1:end-1] = memory[:,2:end]			# forget old memory
		memory[:,end] = get_one_memory_state(p)		# learn new (next) one memory state
		state = get_state_number(memory)
	end
	return state
end


function init_q_matrix()
	# initialize Q-matrices
	Q = Array{Float32,3}(undef, n_states, n_prices, n_agents)
	for i in 1:n_agents
		if q_init_mode[i] == "random"
			Q[:,:,i] = rand(Float32, n_states, n_prices) * expected_coop_profit[i] / (1 - delta[i])
		elseif q_init_mode[i] == "nash"
			Q[:,:,i] = fill(nash_profit[i], n_states, n_prices) / (1 - delta[i])
		elseif q_init_mode[i] == "coop"
			Q[:,:,i] = fill(coop_profit[i], n_states, n_prices) / (1 - delta[i])
		elseif q_init_mode[i] == "expected_payoff"
			idx = [price_numbers for i in 1:n_agents]
			for p in 1:n_prices 
				idx[i] = [p]
				Q[:,p,i] .= mean(getfield.(expected_payoffs[idx...],i)) / (1 - delta[i])
			end
		elseif q_init_mode[i] == "nash_opponents"
			idx = [[p_nash[i]] for i in 1:n_agents]
			for p in 1:n_prices 
				idx[i] = [p]
				Q[:,p,i] .= mean(getfield.(expected_payoffs[idx...],i)) / (1 - delta[i])
			end
		end
	end
	return Q
end


function init_greedy_policy(Q::Array{Float32,3}, alloc_max_idx::Array{Int8,1})
	# initialize strategies
	greedy_policy = Array{Int32, 2}(undef, n_states, n_agents)
	V = Array{Float32,2}(undef, n_states, n_agents)
	for i in 1:n_agents
		for s in 1:n_states
			greedy_policy[s,i] = rand(argmax_(view(Q,s,:,i), alloc_max_idx))
			V[s,i] = Q[s,greedy_policy[s,i],i]
		end
	end
	return V, greedy_policy
end


function init_memory()
	# randomly initialize memory
	if memory_length == 0
		return fill(Int32(1), n_agents, 1)		# this need be a matrix
	else
		return hcat(rand(one_memory_state, memory_length)...)
	end
end