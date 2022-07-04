
function convergence_analysis(last_epoch::Array{Int32,1}, last_memory::Array{Int32,2}, greedy_policy::Array{Int8,3})
	# assess convergence for each session (even non-converged ones) 	
	max_cycle_length = Int32(n_prices)^(n_agents*memory_length) + 1							# longest possible cycle has length n_real_states + 1 (because states are subjective)
	cycle_states = Array{Int32,2}(undef, max_cycle_length, n_sessions)
	cycle_prices = Array{Int8,3}(undef, max_cycle_length, n_agents, n_sessions)
	cycle_profits = Array{NTuple{n_agents,Float32},2}(undef, max_cycle_length, n_sessions)
	cycle_length = Array{Int32,1}(undef, n_sessions)
	profit_gains = Array{Array{Float32,2},1}(undef, n_sessions)

	for z in 1:n_sessions
		last_memory[:,z] = compute_cycles(last_memory[:,z], greedy_policy[:,:,z], view(cycle_states,:,z), view(cycle_prices,:,:,z), view(cycle_profits,:,z), view(cycle_length,z), max_cycle_length)	
		profit_gains[z] = compute_profit_gains(cycle_profits[:,z], cycle_length[z])
	end
	converged = [last_epoch[i] != max_epochs for i in 1:n_sessions]                     # save converged flags

	return converged, count(converged), cycle_states, cycle_prices, cycle_profits, cycle_length, profit_gains
end


function compute_cycles(memory, greedy_policy, cycle_states, cycle_prices, cycle_profits, cycle_length, max_cycle_length)
	# compute states, prices and profits at convergence: arguments are passed by reference to reduce allocations
	state = get_state_number(memory)
	cycle_start = undef
	for t in 1:max_cycle_length															# longest possible cycle has length n_states + 1
		p = greedy_policy[state,:]														# apply optimal greedy_policy
		cycle_start = isvisited(view(cycle_states,1:t-1), state)						# returns when visited if visited
		if cycle_start != undef                                                			# break if state already visited
			cycle_length[:] .= t - cycle_start
			break
		end 
		cycle_states[t] = state
		cycle_prices[t,:] = p
		cycle_profits[t] = expected_payoffs[get_tuple(p,n_agents)...]                  
		state = get_next_state(memory, p)                                           	# state <- next_state
	end
	if cycle_start != 1                                                         		# some sessions do not end at convergence
		start_ = cycle_start
		end_ = min(start_ + cycle_start - 1, max_cycle_length)                   		# overwerite with zeros non cycling states/prices/profits
		cycle_states[1:end_-start_+1] =  cycle_states[start_:end_] 
		cycle_prices[1:end_-start_+1,:] = cycle_prices[start_:end_,:]
		cycle_profits[1:end_-start_+1] = cycle_profits[start_:end_]
	end
	return memory 																		# put memory at convergence (useful when cycle_start != 1)
end


function isvisited(cycle_states, state)
	for t in eachindex(cycle_states)
		cycle_states[t] == state && return t
	end
	return undef
end


function compute_profit_gains(cycle_profits, cycle_length)
	# compute profit gains and price markup over agents and sessions
	profit_gains = Array{Float32,2}(undef, cycle_length, n_agents)
	for t in 1:cycle_length
		profit_gains[t,:] = (cycle_profits[t] .- expected_nash_profit) ./ (expected_coop_profit - expected_nash_profit)
	end
	return profit_gains
end