

function Q_error_analysis(Q::Array{Float32,4}, greedy_policy::Array{Int8,3}, converged::Array{Bool}, cycle_prices, cycle_states, cycle_length)
	n_agents == 2 || return NaN, NaN, NaN, NaN
	greedy_policy_corrected = Array{Int8,3}(undef, n_states, n_agents, n_sessions)
	Q_corrected = Array{Float32,4}(undef, n_states, n_prices, n_agents, n_sessions)
	Threads.@threads for z in 1:n_sessions
		Q_corrected[:,:,:,z], greedy_policy_corrected[:,:,z] = get_corrected_Q(view(Q,:,:,:,z), view(greedy_policy,:,:,z), view(Q_corrected,:,:,:,z), view(greedy_policy_corrected,:,:,z),
																								cycle_prices, cycle_states, cycle_length)
	end
	Q_error = (Q - Q_corrected) ./ Q_corrected
	on_path_Q_loss, on_path_Q_error, on_path_policy_error = get_on_path_errors(Q_corrected, Q_error, greedy_policy, greedy_policy_corrected, cycle_states, cycle_length)
	
	# average percent Q-error in absolute value on equilibrium states
	mean_on_path_Q_error = mean_se(mean.(on_path_Q_error))[1]
	# share of convergence state-action pairs where an agent would be better off by deviating
	mean_on_path_policy_error = mean(mean.(on_path_policy_error))
	# average percent forgone payoff (Q_loss) by acting suboptimally in convergence states (that is, when there is a policy_error)
	# this indicates in percentage terms, how much the agent could have earned by acting optimally (these are profitable deviations!)
	mean_on_path_Q_loss = mean_se(mean.(on_path_Q_loss)[mean.(on_path_Q_loss).>0])[1]

	# same thing, but averagin across cycle lengths groups
	cycle_mean_on_path_Q_error = [mean(mean.(on_path_Q_error)[cycle_length .== x,:]) for x in 1:maximum(cycle_length)]	
	cycle_mean_on_path_policy_error = [mean(mean.(on_path_policy_error)[cycle_length .== x,:]) for x in 1:maximum(cycle_length)]
	cycle_mean_on_path_Q_loss = [mean(mean.(on_path_Q_loss)[mean.(on_path_Q_loss).>0 .&& cycle_length .== x,:]) for x in 1:maximum(cycle_length)]	

	return Q_error, (mean_on_path_Q_error, cycle_mean_on_path_Q_error), (mean_on_path_policy_error, cycle_mean_on_path_policy_error), (mean_on_path_Q_loss, cycle_mean_on_path_Q_loss)
end


function get_corrected_Q(Q, greedy_policy, Q_corrected, greedy_policy_corrected, cycle_prices, cycle_states, cycle_length)
	# compute real Q according to policies at convergence (only works w/ one period memory)
	# given a tuple of fixed policies at convergence, the MDP is fully determined
	# this function computes the real stream of payoffs of playing action a in state s 
	# and playing accortding according to the policy thereafter, while the opponent always plays according to its policy
	max_cycle_length = n_states+1										# after n_real_states + 1 agents must visit one already visited (non coarse) state		
	shock_prices = Array{Float32,2}(undef, maximum(cycle_length)+dev_length-1, n_agents)
	shock_profits = Array{Float32,2}(undef, maximum(cycle_length)+dev_length-1, n_agents)	
	post_prices = Array{Float32,2}(undef, max_cycle_length, n_agents)
	post_profits = Array{Float32,2}(undef, max_cycle_length, n_agents)
	visited_states = Array{Int32,1}(undef, max_cycle_length)			
	for i in 1:n_agents
		for memory in one_memory_state
			for dev_p in 1:n_prices
				instant_profit, pre_cycle_profits, cycle_profits, cycle_length = gen_individual_ir(Int32.([memory]), greedy_policy, i, dev_p, 1, 
														shock_prices, shock_profits, post_prices, post_profits, visited_states, dev_length = 1)[4:end]
				Q_corrected[memory,dev_p,i] = get_d_profits(instant_profit[:], pre_cycle_profits, cycle_profits, cycle_length)[i]
			end
		end	
	end
	for i in 1:n_agents
		for s in 1:n_states
			greedy_policy_corrected[s,i] = minimum(argmax(view(Q_corrected,s,:,i)))
		end
	end
	return Q_corrected, greedy_policy_corrected
end


function get_on_path_errors(Q_corrected, Q_error, greedy_policy, greedy_policy_corrected, cycle_states, cycle_length)
	# compute errors at convergence state-actions pairs (only mem1)
	# look at the percentage difference between Q(s,a) and the true value for those pairs
	# and look if a deviation from what has been learned would be profitable	
	# it probably does not make sense to check error on state-action path
	# because there might be small errors in the visited state-action 
	# but large error in the same state but different actions pairs leading to different greedy policy
	# Note: this analysis does not account for the fact that agents learn their opponents 
	# change their strategy depending on what actions they chose. An action might seem profitable when the 
	# rivals' policy is fixed, but when is not, maybe repeating an action induces the opponent to change their
	# thus leading to a inferior rewards 
	on_path_Q_error = [zeros(cycle_length[z], n_agents) for z in 1:n_sessions]
	on_path_policy_error = [zeros(cycle_length[z], n_agents) for z in 1:n_sessions]
	on_path_Q_loss = [zeros(cycle_length[z], n_agents) for z in 1:n_sessions]
	for z in 1:n_sessions
		for i in 1:n_agents
			for t in 1:cycle_length[z]
				s = cycle_states[t,z]
				# average percentage change between Q and corrected Q at convergence states
				on_path_Q_error[z][t,i] = mean(abs.(Q_error[s,:,i,z])) 				#use cycle_price here in case one wants state-action path
				# count if greedy policy differs from corrected greedy policy
				on_path_policy_error[z][t,i] += count(!isequal(greedy_policy_corrected[s,i,z], greedy_policy[s,i,z])) / cycle_length[z]
				# average percentage forgone payoff in session: positive forgone payoff if greedy_policy != greedy_policy_corrected and zero otherwise
				p = greedy_policy[s,i,z]										# cycle_prices
				p_ = greedy_policy_corrected[s,i,z]								# optimal price in each t of cycle
				on_path_Q_loss[z][t,i] = (Q_corrected[s,p_,i,z] - Q_corrected[s,p,i,z]) / Q_corrected[s,p,i,z]
			end
		end
	end
	return on_path_Q_loss, on_path_Q_error, on_path_policy_error
end



