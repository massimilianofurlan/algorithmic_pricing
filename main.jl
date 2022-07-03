using Base.Threads
using TOML
using Statistics
using ProgressMeter
using Random

include("learning_simulation.jl")
include("functions.jl")
include("io_utils.jl")

run(`rm -rf output`)
run(`mkdir output`)

Random.seed!(231195)

println("initializing global variables...")
# parse configurations from file 
const configfile = TOML.parsefile("config.toml")
const config = configfile["two_agents"]
# input validation
#input_validation(config["a"],config["c"])			# firms must be symmetric Anderson and De Palma

# demand and agents
const a = Float32.(config["a"])						
const c = Float32.(config["c"])
const a0 = Float32.(config["a0"])
const mu = Float32(config["mu"])
const n_agents = length(a)
const n_markets = length(a0)

# Q-agents
const alpha = Float32.(config["alpha"])
const delta = Float32.(config["delta"])
const delta_ = config["delta"]
const q_init_mode = config["q_init_mode"]

# experiment
const n_prices = Int8(config["n_prices"])
const p_nash = config["p_nash"]	 				# lowest nash price is the (p-nash)-th price
const p_coop = n_prices .- p_nash .+ 1			# highest coop price is the (p-coop)-th price
const price_numbers = Int8.(collect(1:n_prices))
const memory_length = config["memory_length"]
const one_memory_n_states = Int32(n_prices)^n_agents
const n_states = one_memory_n_states^memory_length

const n_sessions = config["n_sessions"]
const max_epochs = config["max_epochs"]
const n_episodes = config["n_episodes"]
const convergence_target = config["convergence_target"]
const max_cycle_length = n_states + 1																			# after n_real_states + 1 agents must visit one already visited (non coarse) state	

const beta = ifelse(any(config["intensity"] .== 0.0), config["beta"],  1 .- 1 ./ (config["intensity"]*n_states*n_prices))
const intensity = Float32.(1 ./ ((1 .- beta) * n_states * n_prices))
const eps0 = config["eps0"]
const epsilon = [[eps0 .* beta.^(t-1) for t in (T-1)*n_episodes+1:T*n_episodes] for T in 1:max_epochs]

# impulse responses
const ir_ext = config["ir_ext"]
const dev_type = config["dev_type"]
const dev_length = config["dev_length"]

# compute equilibrium prices and profits
const nash_price, coop_price = gen_equilibrium_prices()
const nash_profit, coop_profit = gen_equilibrium_profits()
const expected_nash_profit = mean(nash_profit, dims = 2)
const expected_coop_profit = mean(coop_profit, dims = 2)

# compute discrete set of prices and payoff matrices
const prices = gen_prices()
const payoffs, expected_payoffs = gen_payoff_matrix()
const one_memory_state = gen_one_memory_states()
const state_number = gen_state_numbers()

show_experiment_details()

include("convergence_analysis.jl")
include("impulse_response.jl")
include("Q_error.jl")

function begin_simulation()
	last_memory = Array{Int32,2}(undef, max(1,memory_length), n_sessions)
	last_epoch = Array{Int32,1}(undef, n_sessions)
	greedy_policy = Array{Int8,3}(undef, n_states, n_agents, n_sessions)
	Q = Array{Float32,4}(undef, n_states, n_prices, n_agents, n_sessions)
	epoch_profits = Array{Float32,3}(undef, max_epochs, n_agents, n_sessions)

	println("running the experiment...")	
	progress = Progress(n_sessions, color=:white, showspeed=true)
	@time Threads.@threads for z in 1:n_sessions
		last_memory[:,z], last_epoch[z], greedy_policy[:,:,z], Q[:,:,:,z], epoch_profits[:,:,z] = learning_simulation(z)
		next!(progress)
	end

	println("analyzing results...")
	@time converged, n_converged, cycle_states, cycle_prices, cycle_profits, cycle_length, profit_gains = convergence_analysis(last_epoch, last_memory, greedy_policy)
	println("computing impulse reponses...")
	@time indiv_ir_price, aggr_ir_price, dev_gains, mean_dev_gains, returned_to_cycle = gen_impulse_response(last_memory, greedy_policy, cycle_prices, cycle_states, cycle_profits, cycle_length, converged)
	println("computing deviation gains...")
	@time dev_gains_matrix, returned_to_cycle_matrix, share_unprofitable_matrix = gen_deviation_matrix(last_memory, greedy_policy, cycle_prices, cycle_states, cycle_profits, cycle_length, converged)
	println("computing errors on equilibrium path...")
 	@time Q_error, on_path_Q_errors, on_path_policy_errors, on_path_Q_losses = Q_error_analysis(Q, greedy_policy, converged, cycle_prices, cycle_states, cycle_length)

	txt_write(converged, last_epoch, profit_gains, mean_dev_gains, cycle_length, returned_to_cycle,on_path_Q_errors, on_path_policy_errors, on_path_Q_losses)
	tex_write(dev_gains_matrix, returned_to_cycle_matrix, share_unprofitable_matrix)
	save_results(last_memory, last_epoch, greedy_policy, Q, epoch_profits, converged, 
				cycle_states, cycle_prices, cycle_profits, cycle_length, profit_gains,
				indiv_ir_price, aggr_ir_price, dev_gains, returned_to_cycle)
end

results = begin_simulation()

if hang_question("should I generate the plots? (n,y): ")
	include("plots.jl")
	@time generate_tikz_plots(results = results)
end
if hang_question("should I store the results? (n,y): ")
	@time jld_write(config, nash_price, coop_price, nash_profit, coop_profit, prices, payoffs, results)
end

