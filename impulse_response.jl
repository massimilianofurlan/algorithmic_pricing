
function gen_impulse_response(last_memory::Array{Int32,2}, greedy_policy::Array{Int8,3}, cycle_prices::Array{Int8,3}, cycle_states::Array{Int32,2}, cycle_profits::Array{NTuple{n_agents,Float32},2}, cycle_length::Array{Int32,1}, converged::Array{Bool}; dev_type = dev_type)
	# generate impulse responses = [pre, shock, post, new_cycle] as ir_length x n_agents arrays
	# pre -> convergence prices/profits, shock -> exogenous shock prices/profits, 
	# post -> response to shock, new_cycle -> new convergence cycle (this may well be the same cycle as pre)
	# impulse responses parts are arrays of dimension n_sessions,                                      		-> even non converged session are analyzed
	# whose elements contain a matrix of dimension n_agents x cycle_length[z],                          	-> exogenous deviation by each agent in each episode of convergence cycle
	# whose elements contain a matrix of dimension ir_(shock/post/new_cycle)_length x n_agents          	-> each row is an episode of the ir, each column is an agent
	max_cycle_length = Int32(n_prices)^(n_agents*memory_length) + 1											# after n_real_states + 1 agents must visit one already visited (non coarse) state	
	shock_prices_ = Array{Float32,2}(undef, maximum(cycle_length)+dev_length-1, n_agents)
	shock_profits_ = Array{Float32,2}(undef, maximum(cycle_length)+dev_length-1, n_agents)	
	post_prices_ = Array{Float32,2}(undef, max_cycle_length, n_agents)
	post_profits_ = Array{Float32,2}(undef, max_cycle_length, n_agents)
	visited_states_ = Array{Int32,1}(undef, max_cycle_length)	
	pre_prices = Array{Matrix{Float32},1}(undef, n_sessions)
	pre_profits = Array{Matrix{Float32},1}(undef, n_sessions)
	shock_prices = [Array{Matrix{Float32},2}(undef, cycle_length[z], n_agents) for z in 1:n_sessions]  
	shock_profits = [Array{Matrix{Float32},2}(undef, cycle_length[z], n_agents) for z in 1:n_sessions]
	post_prices = [Array{Matrix{Float32},2}(undef, cycle_length[z], n_agents) for z in 1:n_sessions]
	post_profits = [Array{Matrix{Float32},2}(undef, cycle_length[z], n_agents) for z in 1:n_sessions]
	new_cycle_prices = [Array{Matrix{Float32},2}(undef, cycle_length[z], n_agents) for z in 1:n_sessions]
	new_cycle_profits = [Array{Matrix{Float32},2}(undef, cycle_length[z], n_agents) for z in 1:n_sessions] 
	indiv_ir_price = [Array{Matrix{Float32},2}(undef, cycle_length[z], n_agents) for z in 1:n_sessions]		# individual impulse response (as opposed to aggregate)
	indiv_ir_profits = [Array{Matrix{Float32},2}(undef, cycle_length[z], n_agents) for z in 1:n_sessions] 	# individual impulse response (as opposed to aggregate)
	returned_to_cycle = [Array{Bool,2}(undef, cycle_length[z], n_agents) for z in 1:n_sessions]  			# true if agents return to previous cycle after shock
	dev_gains = [Array{Float64,3}(undef, cycle_length[z], n_agents, n_agents) for z in 1:n_sessions]

	for z in 1:n_sessions
		# pre-shock analysis
		pre_prices[z] = [prices[cycle_prices[t,i,z],i] for t in 1:cycle_length[z], i in 1:n_agents]     	# get prices at convergence
		pre_profits[z] = [cycle_profits[t,z][i] for t in 1:cycle_length[z], i in 1:n_agents]            	# get profits at convergence
		# shock and post shock analysis
		for dev_agent in 1:n_agents                                                                     	# loop over shocking agent                   
			for dev_t in 1:cycle_length[z]                                                             		# loop over episodes in cycle
				# get exogenous deviation price (out of equilibrium)
				if dev_type in 1:n_prices
					dev_p = dev_type																		# deviate to dev_type price
				elseif dev_type in -(1:n_prices)
					dev_p = max(greedy_policy[cycle_states[dev_t,z],dev_agent,z] + dev_type, 1)				# decrease price by dev_type units 
				elseif dev_type == 0
					dev_p = get_static_best_response(greedy_policy[:,:,z], cycle_states[dev_t,z], dev_agent)# deviate to static BR to future opponent price
				elseif dev_type > n_prices
					dev_p = get_dynamic_best_response(greedy_policy[:,:,z], last_memory[:,z], dev_agent, dev_t, 
														shock_prices_, shock_profits_, post_prices_, post_profits_, visited_states_)
				elseif dev_type < -n_prices
					dev_p = greedy_policy[cycle_states[dev_t,z],dev_agent,z]						# no deviation
				end
				# generate impulse response for (all, even non-converged) session z, with deviating agent dev_agent and at episode dev_t of convergence cycle
				shock_prices[z][dev_t,dev_agent], post_prices[z][dev_t,dev_agent], new_cycle_prices[z][dev_t,dev_agent],
				shock_profits[z][dev_t,dev_agent], post_profits[z][dev_t,dev_agent], new_cycle_profits[z][dev_t,dev_agent], 
				new_cycle_length = gen_individual_ir(last_memory[:,z], greedy_policy[:,:,z], dev_agent, dev_p, dev_t,
														shock_prices_, shock_profits_, post_prices_, post_profits_, visited_states_)
				# pack together impulse response pieces
				indiv_ir_price[z][dev_t,dev_agent] = build_individual_ir(pre_prices[z], shock_prices[z][dev_t,dev_agent], 
																			post_prices[z][dev_t,dev_agent], new_cycle_prices[z][dev_t,dev_agent], ir_ext = ir_ext)
				indiv_ir_profits[z][dev_t,dev_agent] = build_individual_ir(pre_profits[z], shock_profits[z][dev_t,dev_agent], 
																			post_profits[z][dev_t,dev_agent], new_cycle_profits[z][dev_t,dev_agent])
				# compute deviation gains
				dev_gains[z][dev_t,dev_agent,:] = get_deviation_gains(circshift(pre_profits[z],-(dev_t-1)), shock_profits[z][dev_t,dev_agent][dev_t:end,:], post_profits[z][dev_t,dev_agent],
																		new_cycle_profits[z][dev_t,dev_agent], cycle_length[z], new_cycle_length)
				# check if agents return to pre cycle
				returned_to_cycle[z][dev_t,dev_agent] = is_returned_to_cycle(pre_prices[z], new_cycle_prices[z][dev_t,dev_agent], cycle_length[z], new_cycle_length)
			end
		end
	end

	# get average deviation profitability over all dev_t and dev_agents for each session 
	mean_dev_gains = [mean(dev_gains[z][dev_t,dev_agent,dev_agent] for dev_agent in 1:n_agents, dev_t in 1:cycle_length[z]) for z in 1:n_sessions]				# over each dev_t and dev_agent for each session
	#mean_non_dev_gains = [mean(dev_gains[z][dev_t,dev_agent,1:end.!=dev_agent] for dev_agent in 1:n_agents, dev_t in 1:cycle_length[z]) for z in 1:n_sessions]	# over each dev_t and dev_agent for each session
	
	# get aggregate impulse response, over all converged sessions, deviating agents and episode in covergence cycles
	aggr_ir_price, sd_ir_price = build_aggregate_ir(pre_prices, shock_prices, post_prices, new_cycle_prices, cycle_length, converged)
	#aggr_ir_profit, sd_ir_profit = build_aggregate_ir(pre_profits, shock_profits, post_profits, new_cycle_profits, cycle_length, converged)

	return indiv_ir_price, aggr_ir_price, dev_gains, mean_dev_gains, returned_to_cycle
end


function get_static_best_response(greedy_policy, state, dev_agent)
	# compute dev_agent static best response given opponents' convergence strategies and current state
	# static best response maximizes the instantaneous payoff given opponent prices
	profits = Array{Float32}(undef, n_prices);
	p = greedy_policy[state,:]                    					# get the greedy price in state state
	for j in 1:n_prices
		p[dev_agent] = j                                        	# dev_agent tries all the prices 
		profits[j] = get_expected_profits(p)[dev_agent]          	# and looks at the profits she obtains given her opponent plays according to her greedy policy 
	end
	return minimum(argmax(profits))                            		# get lowest best price    
	#greedy_policy[cycle_states[dev_t,z],dev_agent,z] 				#no deviation, testing purposes
end


function get_dynamic_best_response(greedy_policy, memory, dev_agent, dev_t, shock_prices, shock_profits, post_prices, post_profits, visited_states; dev_length = dev_length)
	# compute the true Q(s,:) given agents' strategies and do argmax Q(s,:)
	d_profits = Array{Float32}(undef, n_prices);	
	for dev_p in 1:n_prices
		shock_profits, pre_cycle_profits, cycle_profits, 
		cycle_length = gen_individual_ir(copy(memory), greedy_policy, dev_agent, dev_p, dev_t, shock_prices, shock_profits, 
											post_prices, post_profits, visited_states, dev_length = 1)[4:end]
		d_profits[dev_p] = get_d_profits(shock_profits[dev_t:end,:], pre_cycle_profits, cycle_profits, cycle_length)[dev_agent]
	end
	return argmax(d_profits)														# get lowest best price    
end


function get_d_profits(instant_profit, pre_cycle_profits, cycle_profits, cycle_length)
	# compute discounted stream of rewards: instantaneous reward + stream of reward + infite cycle rewards 
   pre_cycle_length = size(pre_cycle_profits,1)
   d_profits = instant_profit[:] + sum(pre_cycle_profits[t,:] .* delta_.^t for t in 1:pre_cycle_length; init=zeros(Float32,n_agents))
   d_profits += delta_.^pre_cycle_length .* sum(cycle_profits[t,:] .* delta_.^t for t in 1:cycle_length) ./ (1 .- delta_.^cycle_length)
   return d_profits                                                                                                                                                      
end


function gen_individual_ir(memory, greedy_policy, dev_agent, dev_p, dev_t, 
							shock_prices, shock_profits, post_prices, post_profits, visited_states; dev_length = dev_length)
	# generate impulse response for a given deviating agent deviating at the dev_t-the episode of convergence cycle
	max_cycle_length = Int32(n_prices)^(n_agents*memory_length) + 1					# after n_real_states + 1 agents must visit one already visited (non coarse) state
	shock_length = dev_t + dev_length - 1                                     		# agent reaches dev_t, then shocks for dev_length episodes
	# shock period
	state = get_state_number(memory)                                           		# resume session
	for t in 1:shock_length
		p = greedy_policy[state,:]                                					# get best prices given convergence strategies
		if t >= dev_t                                                          		# deviating from dev_t over
			p[dev_agent] = dev_p                                               		# deviating agent choses deviating price
		end
		shock_prices[t,:] = [prices[p[i],i] for i in 1:n_agents]
		shock_profits[t,:] = get_expected_profits(p)
		state = get_next_state(memory, p)                                      		# state <- next_state
	end
	# post-shock period
	post_end, post_length = undef, undef
	for t in 1:max_cycle_length
		p = greedy_policy[state,:]                                					# apply optimal greedy_policy
		post_prices[t,:] = [prices[p[i],i] for i in 1:n_agents]
		post_profits[t,:] = get_expected_profits(p)   	
		state = get_next_state(memory, p)                                      		# state <- next_state		
		post_end = isvisited(view(visited_states,1:t-1), state)						# returns when visited if visited
		if post_end != undef                                                		# break if state already visited
			post_length = t
			break
		end     
		visited_states[t] = state													
	end

	shock_prices_ = view(shock_prices,1:shock_length,:)
	post_prices_ = view(post_prices,1:post_end,:)									# post-shock does not include new cycling prices
	new_cycle_prices_ = view(post_prices,post_end+1:post_length,:)    				# extract post_prices the new cycling prices
	shock_profits_ = view(shock_profits,1:shock_length,:)
	post_profits_ = view(post_profits,1:post_end,:)									# post-shock does not include new cycling prices
	new_cycle_profits_ = view(post_profits,post_end+1:post_length,:)    			# extract post_prices the new cycling prices
	new_cycle_length = post_length - post_end

	return shock_prices_, post_prices_, new_cycle_prices_, 
			shock_profits_, post_profits_, new_cycle_profits_, new_cycle_length
end


function build_individual_ir(pre, shock, post, new_cycle; ir_ext = 1)
	# pre, shock, post, new_cycle do not have redundant information 
	# to print impulse response, extend the pre-shock episodes and post-shock (cycling) episode
	pre_ext = vcat([pre for i in 1:ir_ext]...)
	new_cycle_ext = vcat([new_cycle for i in 1:ir_ext]...)
	return vcat(pre_ext, shock, post, new_cycle_ext)
end


function is_returned_to_cycle(pre_price, new_cycle_price, cycle_length, new_cycle_length)
	# check if new_cycle_price for individual impulse (given z, dev_agent, dev_t) equals previous price cycle of session z
	new_cycle_length != cycle_length && return false
	for t in 1:new_cycle_length
		isequal(new_cycle_price,pre_price) && return true
		new_cycle_price = circshift(new_cycle_price,1)				# cycle do not necessairly start at the same price, circshift prices anc check again
	end
	return false
end


function get_deviation_gains(pre_profits, shock_profits, pre_cycle_profits, new_cycle_profits, cycle_length, new_cycle_length)
	# compute deviation gain / loss
	# the discounted sum of an infinite sum of profits \sum_{t=1}^\infty \delta^{t-1} \pi_t
	# where {\pi_t}_{t=1}^\infty is an infinitely repeated cycle of lenght K starting from t = 1 onward
	# is obtained as d_cycle(K,{\pi}) = ( \sum_{k=1}^K \delta^(t-1) \pi_k ) / 1 - \delta^K
	# then, on the path of equilibrium: d_cycle(cycle_length,pre_profits)
	# and by deviating: \sum_{t=1}^shock_post_len delta^{t-1} shock_post_profits + delta^shock_post_len * d_cycle(new_cycle_length,new_cycle_profits)
	# NOTE: when dev_t > 1 you want to compare profits starting from dev_t onward so that delta^(dev_t-1) = 1
	path_d_profits = sum(pre_profits[t,:] .* delta_.^(t-1) for t in 1:cycle_length) ./ (1 .- delta_.^cycle_length)
	dev_d_profits = get_d_profits(shock_profits, pre_cycle_profits, new_cycle_profits, new_cycle_length)
	return (dev_d_profits - path_d_profits) ./ path_d_profits							# compute deviation (percent) gain/loss for individual impulse
end


function build_aggregate_ir(pre, shock, post, new_cycle, cycle_length, converged; ir_length = 20)
	# build aggregate impulse response for converged sessions only
	# these array comprehensions unpack the impulse response and aggregate them over all sessions, deviating agents, and deviating episode in cycle
	# shock and post arrays have 5 dimensions: [session][deviating_agent,deviating_episode_in_cycle][episode,agent]
	# deviating_agent is the index of the deviating agent
	# deviating_episode_in_cycle is the dev_t-th episode in the cycle in which the deviation ocurrs 
	# [session][deviating_agent,deviating_episode_in_cycle] is a ir_part_length x n_agents
	# all together they form the aggregate impulse response
	# [dev_t,dev_agent][dev_t=end,dev_agent] extracts the price set by the deviating agent in the dev_t-th episode in cycle
	# [dev_t,dev_agent][:,1:end.!=dev_agent] extracts the prices set by the non deviating agents
	n_converged = count(converged)
	n_converged == 0 && return undef, undef												# early return if no session has converged
	converged_sessions = (1:n_sessions)[converged]

	mean_pre = Array{Float32,1}(undef, n_converged)
	mean_shock_dev = Array{Array{Float32,1},1}(undef, n_converged)
	mean_shock_non_dev = Array{Array{Float32,1},1}(undef, n_converged)
	mean_post_dev = Array{Array{Array{Float32,1},2},1}(undef, n_converged)
	mean_post_non_dev = Array{Array{Array{Float32,1},2},1}(undef, n_converged)
	mean_cycle_dev = Array{Array{Array{Float32,1},2},1}(undef, n_converged)
	mean_cycle_non_dev = Array{Array{Array{Float32,1},2},1}(undef, n_converged)
	mean_post_cycle_dev = Array{Array{Float32,2},1}(undef, n_converged)
	mean_post_cycle_non_dev = Array{Array{Float32,2},1}(undef, n_converged)
 	post_cycle_dev = [Array{Float32,3}(undef, cycle_length[z], n_agents, ir_length) for z in converged_sessions]	
	post_cycle_non_dev = [Array{Float32,3}(undef, cycle_length[z], n_agents, ir_length) for z in converged_sessions]	
	mean_post_cycle_dev = Array{Float32,2}(undef, ir_length, n_converged)
	mean_post_cycle_non_dev = Array{Float32,2}(undef, ir_length, n_converged)
	for k in 1:n_converged
		z = converged_sessions[k]
		mean_pre[k] = mean(pre[z])
		mean_shock_dev[k] = mean([shock[z][dev_t,dev_agent][dev_t:end,dev_agent] for dev_t in 1:cycle_length[z], dev_agent in 1:n_agents])
		mean_shock_non_dev[k] = mean(mean([shock[z][dev_t,dev_agent][dev_t:end,1:end.!=dev_agent] for dev_t in 1:cycle_length[z], dev_agent in 1:n_agents]), dims = 2)[:,1]
		mean_post_dev[k] = [post[z][dev_t,dev_agent][:,dev_agent] for dev_t in 1:cycle_length[z], dev_agent in 1:n_agents]
		mean_post_non_dev[k] = [mean(post[z][dev_t,dev_agent][:,1:end.!=dev_agent],dims=2)[:] for dev_t in 1:cycle_length[z], dev_agent in 1:n_agents]
		mean_cycle_dev[k] = [new_cycle[z][dev_t,dev_agent][:,dev_agent] for dev_t in 1:cycle_length[z], dev_agent in 1:n_agents]
		mean_cycle_non_dev[k] = [mean(new_cycle[z][dev_t,dev_agent][:,1:end.!=dev_agent],dims=2)[:] for dev_t in 1:cycle_length[z], dev_agent in 1:n_agents]
		for dev_agent in 1:n_agents
			for dev_t in 1:cycle_length[z]
				len = min(length(mean_post_dev[k][dev_t,dev_agent]),ir_length)
				post_cycle_dev[k][dev_t,dev_agent,1:len] = mean_post_dev[k][dev_t,dev_agent][1:len]
				post_cycle_dev[k][dev_t,dev_agent,len+1:ir_length] = vcat([mean_cycle_dev[k][dev_t,dev_agent] for i in 1:ir_length]...)[1:ir_length-len]
				post_cycle_non_dev[k][dev_t,dev_agent,1:len] = mean_post_non_dev[k][dev_t,dev_agent][1:len]
				post_cycle_non_dev[k][dev_t,dev_agent,len+1:ir_length] = vcat([mean_cycle_non_dev[k][dev_t,dev_agent] for i in 1:ir_length]...)[1:ir_length-len]
			end
		end
		mean_post_cycle_dev[:,k] = mean(post_cycle_dev[k][dev_t,dev_agent,:] for dev_t in 1:cycle_length[z], dev_agent in 1:n_agents)
		mean_post_cycle_non_dev[:,k] = mean(post_cycle_non_dev[k][dev_t,dev_agent,:] for dev_t in 1:cycle_length[z], dev_agent in 1:n_agents)
	end
	# computing aggregate piecies
	aggr_pre, sd_pre = mean(mean_pre), std(mean_pre)
	aggr_shock_dev, sd_shock_dev = mean(mean_shock_dev), std(mean_shock_dev)
	aggr_shock_non_dev, sd_shock_non_dev = mean(mean_shock_non_dev), std(mean_shock_non_dev)	#infinitesimal error compared to aggr_pre
	aggr_post_cycle_dev, sd_post_cycle_dev = mean(mean_post_cycle_dev, dims=2)[:], std(mean_post_cycle_dev,dims=2)
	aggr_post_cycle_non_dev, sd_post_cycle_non_dev = mean(mean_post_cycle_non_dev, dims=2)[:], std(mean_post_cycle_dev,dims=2)
	# gluing pieces
	aggr_ir, aggr_sd = zeros(ir_length,2), zeros(ir_length,2)
	aggr_ir[1:ir_ext,:] .= aggr_pre
	aggr_ir[ir_ext+1:ir_ext+dev_length,:] = [aggr_shock_dev aggr_shock_non_dev]
	aggr_ir[ir_ext+dev_length+1:ir_length,:] = [aggr_post_cycle_dev aggr_post_cycle_non_dev][1:ir_length-ir_ext-dev_length,:]
	aggr_sd[1:ir_ext,:] .= sd_pre
	aggr_sd[ir_ext+1:ir_ext+dev_length,:]  = [sd_shock_dev sd_shock_non_dev]
	aggr_sd[ir_ext+dev_length+1:ir_length,:] = [sd_post_cycle_dev sd_post_cycle_non_dev][1:ir_length-ir_ext-dev_length,:]

	return aggr_ir, aggr_sd
end


function get_Q_errors(Q, last_memory, greedy_policy, cycle_prices, cycle_states, cycle_profits, cycle_length, converged)
	# compute Q errors on path in percentage terms 
	max_cycle_length = Int32(n_prices)^(n_agents*memory_length) + 1	
	shock_prices_ = Array{Float32,2}(undef, maximum(cycle_length)+dev_length-1, n_agents)
	shock_profits_ = Array{Float32,2}(undef, maximum(cycle_length)+dev_length-1, n_agents)	
	post_prices_ = Array{Float32,2}(undef, max_cycle_length, n_agents)
	post_profits_ = Array{Float32,2}(undef, max_cycle_length, n_agents)
	visited_states_ = Array{Int32,2}(undef, max_cycle_length, n_agents)	
	on_path_Q_error = zeros(n_agents, n_sessions)
	d_profits = Array{Float64}(undef, n_prices);	
	for z in 1:n_sessions
		for i in 1:n_agents                                                                     		# loop over shocking agent                   
			for t in 1:cycle_length[z]                                                             		# loop over episodes in cycle
				for p in 1:n_prices
					shock_profits, pre_cycle_profits, cycle_profits_, 
					cycle_length_ = gen_individual_ir(copy(last_memory[:,z]), greedy_policy[:,:,z], i, p, t, shock_prices_, shock_profits_, 
														post_prices_, post_profits_, visited_states_, dev_length = 1)[4:end]
					d_profits[p] = get_d_profits(shock_profits[t:end,:], pre_cycle_profits, cycle_profits_, cycle_length_)[i]
				end
				on_path_Q_error[i,z] += mean(abs.((Q[copy(last_memory[z]),:,i,z] - d_profits)./ Q[copy(last_memory[z]),:,i,z])) ./ cycle_length[z]       
			end
		end
	end
	# compute percentage gains from dynamic best response
	dev_gains = gen_impulse_response(last_memory, greedy_policy, cycle_prices, cycle_states, cycle_profits, cycle_length, converged; dev_type = 100)[3]
	# compute share of policy errors on path
	on_path_policy_errors = [count(dev_gains[z][:,dev_agent,dev_agent] .> 1e-10)/(cycle_length[z]) for dev_agent in 1:n_agents, z in (1:n_sessions)[converged]]
	# compute consequential loss on path
	on_path_Q_losses = mean.([dev_gains[z][:,dev_agent,dev_agent] for dev_agent in 1:n_agents, z in (1:n_sessions)[converged]])
	on_path_Q_losses[on_path_Q_losses .<= 1e-10] .= 0 	# machine zero
	# compute nash equilibriums
	is_nash = [all(on_path_policy_errors[:,z] .== 0) for z in 1:count(converged)]

	return on_path_policy_errors, on_path_Q_losses, on_path_Q_error, is_nash
end
