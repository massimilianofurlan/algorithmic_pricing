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
		println(io, "intensity: \t\t " , intensity)
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


function txt_write(converged, last_epoch, profit_gains, mean_dev_gains, cycle_length, returned_to_cycle, on_path_Q_errors, on_path_policy_errors, on_path_Q_losses)
	for sub in [[converged, "converged"], [.!converged, "not_converged"]]
		if count(sub[1]) != 0
			subset, subset_name = sub[1], sub[2]
			share = mean(sub[1])																				# share of converged/non_converged sessions
			avg_last_epoch = mean_se(last_epoch[subset].-convergence_target)[1]														# average number of episodes to reach convergence
			quartile_last_epoch = quantile!(last_epoch[subset].-convergence_target,[0, 0.25, 0.5, 0.75, 1])							# quantiles of the numbers of episodes to reach convergence
			agents_profit_gain = mean_se(mean.(profit_gains[subset], dims = 1))									# get average profit gain for each agent across all (subset) sessions
			overall_profit_gain = mean_se(mean.(profit_gains[subset]))[1]										# get average profit gain across all agents and (subset) sessions
			quartile_profit_gain = quantile!(mean.(profit_gains[subset]),[0, 0.25, 0.5, 0.75, 1]) 				# quantiles of average profit gain for each session across all agent
			aggr_dev_gains = mean_se(mean_dev_gains[subset])[1]													# get average deviation gain/loss from deviation, across each session
			avg_cycle_length = mean_se(cycle_length)[1]
			dist_cycle_length = [count(x->x==i,cycle_length) for i in 1:maximum(cycle_length[subset])]			# get cycle length absolute frequencies
			share_returned_to_cycle = mean(all.(returned_to_cycle[subset]))										# get share of (subset) sessions in which the deviation (for all dev_t and dev_agent) did not breaks convergence
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
				print(io, "average percent Q-error on path: \t", round.(on_path_Q_errors[1][1], digits = 5))
				println(io, "  (", round.(on_path_Q_errors[1][2], digits = 5), ")")
				#println(io, ", ", round.(on_path_Q_errors[2], digits = 5))
				println(io, "share policy errors on path: \t\t", round.(on_path_policy_errors[1], digits = 5))
				#println(io, ", ", round.(on_path_policy_errors[2], digits = 5))
				print(io, "percentage forgone payoff on path: \t", round.(on_path_Q_losses[1][1], digits = 5))
				println(io, "  (", round.(on_path_Q_losses[1][2], digits = 5), ")")
				#println(io, ", ", round.(on_path_Q_losses[2], digits = 5))
			end
		end
	end	
	run(`cat output/experiment_results.txt`)
end


function tex_write(dev_gains_matrix, returned_to_cycle_matrix, share_unprofitable_matrix)
	n_agents == 2 || return nothing
	is_static_best_response(matrix,x,y) = (x < n_prices && y > 1 && argmax(getindex.(expected_payoffs,1)[:,n_prices-y+1]) == n_prices - x )
	is_significant(matrix,x,y) = (x < n_prices && y > 1 && t_dev_gains_table[x,y] <= -1.645)
	is_significant_best_response(matrix,dev_p,pre_p) = (is_significant(matrix,dev_p,pre_p) && is_static_best_response(matrix,dev_p,pre_p))
	is_greater_than(matrix,dev_p,pre_p) = (pre_p != 1 && matrix[dev_p,pre_p] >= 0.95)
	
	prices_string = string.(round.(prices[n_prices:-1:2,1],digits=3))
	header = vcat(" ",prices_string)
	# exclude p_pre = 1 and p_dev = n_prices: both have no thave deviations
	avg_dev_gains_matrix = getindex.(dev_gains_matrix,1)[1:n_prices-1,2:n_prices]
	sd_dev_gains_matrix = getindex.(dev_gains_matrix,2)[1:n_prices-1,2:n_prices]
	t_dev_gains = avg_dev_gains_matrix ./ sd_dev_gains_matrix
	# round and glue table
	avg_dev_gains_table = round.(hcat(prices[n_prices-1:-1:1,1], avg_dev_gains_matrix[end:-1:1,end:-1:1]), digits = 3)
	t_dev_gains_table = round.(hcat(prices[n_prices-1:-1:1,1], t_dev_gains[end:-1:1,end:-1:1]), digits = 3)
	returned_to_cycle_table = round.(hcat(prices[n_prices-1:-1:1,1], returned_to_cycle_matrix[n_prices-1:-1:1,n_prices:-1:2]), digits = 3)
	share_unprofitable_table = round.(hcat(prices[n_prices-1:-1:1,1], share_unprofitable_matrix[n_prices-1:-1:1,n_prices:-1:2]), digits = 3)
	# generate latex tables
	h_br = LatexHighlighter(is_static_best_response, ["textbf"])
	h_gt = LatexHighlighter(is_greater_than, ["textbf"])	
	h_ss = LatexHighlighter(is_significant, ["color{red}"])	
	h_ssbr = LatexHighlighter(is_significant_best_response, ["color{red}", "textbf"])	
	open("output/tables.tex", "w") do io
		println(io, raw"\documentclass[]{article}")
		println(io, raw"\usepackage{xcolor}")
		println(io, raw"\usepackage[landscape]{geometry}")
		println(io, raw"\begin{document}")
		pretty_table(io, avg_dev_gains_table, header = header, highlighters = (h_ssbr, h_ss, h_br), backend = Val(:latex))
		pretty_table(io, share_unprofitable_table, header = header, highlighters = (h_gt), backend = Val(:latex))		
		pretty_table(io, returned_to_cycle_table, header = header, backend = Val(:latex))
		println(io, raw"\end{document}")
	end
	cd("output")
	run(pipeline(`pdflatex -interaction=batchmode tables.tex`, devnull))
	rm("tables.log"), rm("tables.aux")
	cd("..")
end


function mean_se(x; dims = :)
	 # returns a tuple containing mean and standard error of x over dimension dims
	 mean_ = mean(x, dims = dims)
	 se_ = std(x, corrected = false, dims = dims)
	 return [(mean_[i], se_[i]) for i in 1:length(mean_)]
end


function save_results(last_memory, last_epoch, greedy_policy, Q, epoch_profits, converged, 
					cycle_states, cycle_prices, cycle_profits, cycle_length, profit_gains,
					indiv_ir_price, aggr_ir_price, dev_gains, returned_to_cycle)
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
				"returned_to_cycle"=> returned_to_cycle
	)
	return results
end


function jld_write(config, nash_price, coop_price, nash_profit, coop_profit, prices, payoffs, results)
	# write data to .jld 
	println("storing results...")	
	data = Dict(
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