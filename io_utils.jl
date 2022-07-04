using JLD
using PrettyTables

function input_validation(a,c)
	if !all(y->y==a[1], a) || !all(y->y==c[1], c)
		all(y->y==a[1], a) || println("This software only supports symmetric firms: a = $a not supported")
		all(y->y==c[1], c) || println("This software only supports symmetric firms: c = $c not supported")
		exit()
	end
end


function show_experiment_details()
	# save experiment details to file, and show them on terminal
	open("output/experiment_details.txt","w") do io
		println(io, "\n[experiment details]")
		println(io, "git branch: \t\t main")
		println(io, "number of sessions: \t " , n_sessions)
		println(io, "number of episodes: \t " , n_episodes)
		println(io, "max number of epochs: \t " , max_epochs)
		println(io, "convergence target: \t " , convergence_target)
		println(io, "alpha: \t\t\t " , alpha)
		println(io, "beta: \t\t\t " , beta)
		println(io, "nu: \t\t\t " , nu)
		println(io, "eps0: \t\t\t " , eps0)
		println(io, "delta: \t\t\t " , delta_)
		println(io, "init Q matrices: \t " , q_init_mode)
		println(io, "number of agents: \t " , n_agents)
		println(io, "number of prices: \t " , n_prices)
		println(io, "nash price index: \t " , p_nash)
		println(io, "memory length: \t\t " , memory_length)
		println(io, "a: \t\t\t " , a)
		println(io, "c: \t\t\t " , c)
		println(io, "a0: \t\t\t " , a0)
		println(io, "mu: \t\t\t " , mu)
		println(io, "deviation type: \t " , dev_type)
		println(io, "deviation length: \t " , dev_length)
		println(io, "nash price: \t\t " , nash_price)
		println(io, "coop price: \t\t " , coop_price)
		println(io, "nash profit: \t\t " , nash_profit)
		println(io, "coop profit: \t\t " , coop_profit)
		println(io)
	end
	run(`cat output/experiment_details.txt`)
end


function txt_write(converged, last_epoch, profit_gains, mean_dev_gains, cycle_length, returned_to_cycle, on_path_policy_errors, on_path_Q_losses, on_path_Q_error, is_nash)
	for sub in [[converged, "converged"], [.!converged, "not_converged"]]
		count(sub[1]) != 0 || continue
		subset, subset_name = sub[1], sub[2]
		share = mean(sub[1])																				# share of converged/non_converged sessions
		avg_last_epoch = mean_se(last_epoch[subset].-convergence_target)[1]														# average number of episodes to reach convergence
		quartile_last_epoch = quantile!(last_epoch[subset].-convergence_target,[0, 0.25, 0.5, 0.75, 1])							# quantiles of the numbers of episodes to reach convergence
		agents_profit_gain = mean_se(mean.(profit_gains[subset], dims = 1))									# get average profit gain for each agent across all (subset) sessions
		overall_profit_gain = mean_se(mean.(profit_gains[subset]))[1]										# get average profit gain across all agents and (subset) sessions
		quartile_profit_gain = quantile!(mean.(profit_gains[subset]),[0, 0.25, 0.5, 0.75, 1]) 				# quantiles of average profit gain for each session across all agent
		aggr_dev_gains = mean_se(mean_dev_gains[subset])[1]													# get average deviation gain/loss from deviation, across each session
		avg_cycle_length = mean_se(cycle_length[subset])[1]
		dist_cycle_length = [count(x->x==i,cycle_length[subset]) for i in 1:maximum(cycle_length[subset])]			# get cycle length absolute frequencies
		share_returned_to_cycle = mean(all.(returned_to_cycle[subset]))										# get share of (subset) sessions in which the deviation (for all dev_t and dev_agent) did not breaks convergence
		avg_on_path_policy_errors = mean_se(on_path_policy_errors)[1]
		avg_on_path_Q_losses = mean_se(on_path_Q_losses[on_path_Q_losses .> 1e-10])[1]
		avg_on_path_Q_error = mean_se(on_path_Q_error)[1]			
		avg_is_nash = mean(is_nash)[1]			
		open("output/experiment_results.txt","a") do io
			println(io, "\n[experiment results]")
			println(io, "($subset_name)")
			#
			println(io, "share $subset_name sessions: \t\t", share)
			#
			print(io, "average epochs to convergence: \t\t", round(Float64(avg_last_epoch[1]), digits = 2))
			println(io, "  (", round(Float64(avg_last_epoch[2]), digits = 2), ")")
			println(io, "epochs to convergence (quantiles): \t", quartile_last_epoch)
			###
			for i in 1:n_agents
				print(io, "average profit gain of agent $i: \t", round.(Float64(agents_profit_gain[i][1]), digits = 5))
				println(io, "  (", round.(Float64(agents_profit_gain[i][2]), digits = 5), ")")
			end
			print(io, "average profit gain across agents: \t", round.(Float64.(collect(overall_profit_gain[1])), digits = 5))
			println(io, "  (", round.(Float64.(collect(overall_profit_gain[2])), digits = 5), ")")
			println(io, "average profit gains (quantiles): \t", round.(quartile_profit_gain, digits = 5))
			print(io, "average cycle length: \t\t\t", round(Float64(avg_cycle_length[1]), digits = 2))
			println(io, "  (", round(Float64(avg_cycle_length[2]), digits = 2), ")")
			println(io, "cycle length distribution: \t\t", dist_cycle_length/length(cycle_length))
			###
			print(io, "average percent deviation gain: \t", round.(aggr_dev_gains[1], digits = 5))
			println(io, "  (", round.(Float64(aggr_dev_gains[2]), digits = 5), ")")
			println(io, "share returned to cycle: \t\t", round.(share_returned_to_cycle, digits = 5))
			###
			if subset_name == "converged"
				print(io, "average percent Q-error on path: \t", round.(avg_on_path_Q_error[1], digits = 5))
				println(io, "  (", round.(avg_on_path_Q_error[2], digits = 5), ")")
				print(io, "share policy errors on path: \t\t", round.(avg_on_path_policy_errors[1], digits = 5))
				println(io, ", ", round.(avg_on_path_policy_errors[2], digits = 5))
				print(io, "percentage forgone payoff on path: \t", round.(avg_on_path_Q_losses[1], digits = 5))
				println(io, "  (", round.(avg_on_path_Q_losses[2], digits = 5), ")")
				#println(io, ", ", round.(on_path_Q_losses[2], digits = 5))
				print(io, "share of Nash equilibria: \t\t", round.(avg_is_nash, digits = 5))
			end
			println(io)
		end
	end	
	run(`cat output/experiment_results.txt`)
end


function mean_se(x; dims = :)
	 # returns a tuple containing mean and standard error of x over dimension dims
	 mean_ = mean(x, dims = dims)
	 se_ = std(x, corrected = false, dims = dims)
	 return [(mean_[i], se_[i]) for i in 1:length(mean_)]
end


function save_results(last_memory, last_epoch, greedy_policy, Q, epoch_profits, converged, 
					cycle_states, cycle_prices, cycle_profits, cycle_length, profit_gains,
					indiv_ir_price, aggr_ir_price, dev_gains, returned_to_cycle,
					on_path_policy_errors, on_path_Q_losses, on_path_Q_error, is_nash)
	results = Dict(
				"last_memory"=> last_memory, 
				"last_epoch"=> last_epoch, 
				"greedy_policy"=> greedy_policy, 
				"Q"=> Q, 
				"epoch_profits"=> epoch_profits, 
				"converged"=> converged, 
				"cycle_states"=> cycle_states, 
				"cycle_prices"=> cycle_prices, 
				"cycle_profits"=> cycle_profits, 
				"cycle_length"=> cycle_length,
				"profit_gains"=> profit_gains, 
				"indiv_ir_price"=> indiv_ir_price, 
				"aggr_ir_price"=> aggr_ir_price,
				"dev_gains"=> dev_gains,
				"returned_to_cycle"=> returned_to_cycle,
				"on_path_policy_errors" => on_path_policy_errors,
				"on_path_Q_losses" => on_path_Q_losses,
				"on_path_Q_error" => on_path_Q_error,
				"is_nash" => is_nash
	)
	return results
end


function jld_write(config, nash_price, coop_price, nash_profit, coop_profit, prices, payoffs, results)
	# write data to .jld 
	println("storing results...")	
	data = Dict{String,Any}(
				"config"=> config, 
				"nash_price"=> nash_price, 
				"coop_price"=> coop_price, 
				"nash_profit"=> nash_profit, 
				"coop_profit"=> coop_profit, 
				"prices"=> prices, 
				"payoffs"=> payoffs
	)
	save("output/data.jld", merge!(data,results))
end


function jld_read(path)
	# read data from .jld
	config = load("$path.jld", "config")
	nash_price = load("$path.jld", "nash_price")
	coop_price = load("$path.jld", "coop_price")
	nash_profit = load("$path.jld", "nash_profit")
	coop_profit = load("$path.jld", "coop_profit")
	prices = load("$path.jld", "prices")
	payoffs = load("$path.jld", "payoffs")
	last_memory = load("$path.jld", "last_memory")
	last_epoch = load("$path.jld", "last_epoch")
	greedy_policy = load("$path.jld", "greedy_policy")
	Q = load("$path.jld", "Q")
	epoch_profits = load("$path.jld", "epoch_profits")
	converged = load("$path.jld", "converged")
	cycle_states = load("$path.jld", "cycle_states")
	cycle_prices = load("$path.jld", "cycle_prices")
	cycle_profits = load("$path.jld", "cycle_profits")
	cycle_length = load("$path.jld", "cycle_length")
	profit_gains = load("$path.jld", "profit_gains")
	indiv_ir_price = load("$path.jld", "indiv_ir_price")
	aggr_ir_price = load("$path.jld", "aggr_ir_price")
	dev_gains = load("$path.jld", "dev_gains")
	returned_to_cycle = load("$path.jld", "returned_to_cycle")
    return config, nash_price, coop_price, nash_profit, coop_profit, prices, payoffs 
    		last_memory, last_epoch, greedy_policy, Q, epoch_profits, converged, 
			cycle_states, cycle_prices, cycle_profits, cycle_length,
			profit_gains, indiv_ir_price, aggr_ir_price, dev_gains, returned_to_cycle
end


function hang_question(question)
	while true
		print(question)
		line = readline()
		line == "n" && return false
		line != "y" || return true
	end
end