using Statistics
using JLD
using PGFPlotsX

function generate_tikz_plots(;results = undef)
	println("loading results...")
	if in_session == true
		last_epoch, greedy_policy, epoch_profits, converged, 
		cycle_prices, cycle_length, profit_gains, 
		indiv_ir_price, dev_gains, aggr_ir_price = getindex.(Ref(results),
		["last_epoch", "greedy_policy", "epoch_profits", "converged",
		"cycle_prices", "cycle_length", "profit_gains", 
		"indiv_ir_price", "dev_gains", "aggr_ir_price"])
	else
		last_epoch, greedy_policy, epoch_profits, converged, 
		cycle_prices, cycle_length, profit_gains, 
		indiv_ir_price, dev_gains, aggr_ir_price = load("output/data.jld", 
			"last_epoch", "greedy_policy", "epoch_profits", "converged",
			"cycle_prices", "cycle_length", "profit_gains", 
			"indiv_ir_price", "dev_gains", "aggr_ir_price")
	end
	n_converged = sum(converged)
	println("generating tikz plots... (be patient)")
	isdir("output/tikz") || run(`mkdir -p output/tikz`)
	subset = rand((1:n_sessions)[converged], 9)
	open("output/subset.txt","w") do io println(io, subset) end
	tikz_plot_hm(converged, cycle_length, cycle_prices)
	tikz_plot_aggr_ir(n_converged, aggr_ir_price)
	tikz_plot_ir_group(indiv_ir_price, cycle_length, dev_gains, subset)
	tikz_plot_aggr_pg(last_epoch, epoch_profits)
	tikz_plot_pg_group(last_epoch, epoch_profits, subset)
end


function tikz_plot_hm(converged, cycle_length, cycle_prices)
	# plot heatmap of prices visited at convergence (only converged sessions are considered)	
	
	if length(c_part) > 1 && length(c_part) < n_prices
		cuts = zeros(Int64,length(c_part)-1)
		cuts[1] = c_part[1]
		for i in 2:length(c_part)-1
			cuts[i] = cuts[i-1] + c_part[i]
		end
	end

	if n_agents <= 2 												# cannot do an n-dimensional heatmpas with n > 2 
		dims = Tuple([n_prices for i in 1:n_agents])
		heatmap_prices = zeros(Int64,dims)
		for z in collect(1:n_sessions)[converged]
			for t in 1:cycle_length[z]
				p = cycle_prices[t,:,z]						
				heatmap_prices[p...] += 1 							# count greedy_policy profiles absolute frequency 
			end
		end
		x = repeat(price_numbers, outer = n_prices)
		y = repeat(price_numbers[end:-1:1], inner = n_prices)
		meta = vec(heatmap_prices)
		coord = Coordinates(x, y; meta = meta)
		axis = @pgf Axis(
		    {
		    	xlabel= raw"Agent $1$",
		    	ylabel= raw"Agent $2$", 
		        enlargelimits = false,
		        xtick = price_numbers,
		        ytick = price_numbers,
		        yticklabels = [string(i) for i in n_prices:-1:1],
		        "colorbar horizontal",
		        "colormap/jet",
		    },
		    PlotInc(
		        {
		            matrix_plot,
		            mark = "",
		            point_meta = "explicit",
		            "mesh/cols" = n_prices
		        },
		        coord,
		    ),
		    [raw"\draw[thick, red, dashed] (axis cs: 2.5,13.5) rectangle (axis cs: 1.5,14.5);"],
        	[raw"\draw[thick, red, dashed] (axis cs: 13.5,2.5) rectangle (axis cs: 14.5,1.5);"]
		)
		if length(c_part) > 1 && length(c_part) < n_prices
			@pgf for cut in cuts
			    pl = Plot(
			    	{
			            color = "green!30!lightgray",
			            dashed
			        },
			        Coordinates([(0.5, cut+0.5),(15.5, cut+0.5)]),
			    ) 	
				push!(axis, pl)
				pl = Plot(
			    	{
			            color = "green!30!lightgray",
			            dashed
			        },
			        Coordinates([(cut+0.5,0.5),(cut+0.5,15.5)])
			    ) 	
				push!(axis, pl)
			end
		end
		pgfsave("output/hm.pdf", axis)
		pgfsave("output/tikz/hm.tikz", axis)
	end
end


function tikz_plot_aggr_ir(n_converged, aggr_ir_price)
	n_converged == 0 && return 
	expected_nash_price = mean(nash_price, dims = 2)
	expected_coop_price = mean(coop_price, dims = 2)
	# https://ctan.mirror.garr.it/mirrors/ctan/graphics/pgf/contrib/pgfplots/doc/pgfplots.pdf#page=194
	# red, gree, blue, cyab, magenta, yellow, black. white, darkgray, ligthgray, brown, lime, olive, orange, pink, purple, teal, violet 
	col = ["red","blue",  "purple", "orange"]
	sha = ["triangle", "square",]#triangle*
	lab = ["defector", "compliant",]
	siz = ["3pt","2.5pt"]
	axis = @pgf Axis(
	    {
	        xlabel = raw"$t$", ylabel = raw"$p_t$", 
	        y_label_style = "{rotate=-90}",
			title = "",
			xtick = [ir_ext,ir_ext+1,collect(ir_ext+5:5:30)...],			
			xticklabels = ["0", "1", "5", "10", "15", "20", "25", "30"],			
			ytick = [expected_nash_price...,mean(aggr_ir_price[1:ir_ext,:]),expected_coop_price...],
	       	ymin = minimum(prices)-0.01, ymax = maximum(prices)+0.01,
	       	xmin = 0.8, xmax = 20.2,
			legend_pos="south east",
	    },
	    HLine({ dashed, black!30!darkgray }, expected_nash_price[1]),
	    HLine({ dashed, black!30!darkgray }, expected_coop_price[1]),
	    HLine({ dashed, gray }, mean(aggr_ir_price[1:ir_ext,:])),
		VLine({ dashed, gray }, ir_ext+1),
	);
	@pgf for i in 2:-1:1
	    pl = Plot(
	    	{
	            color = col[i],
	            mark  = sha[i], 
				mark_size = siz[i],
				thick
	        },
	        Table(
	            x = 1:20,
	            y = aggr_ir_price[:,i],
	        )
	    ) 	
	    push!(axis, pl)
	    push!(axis, LegendEntry(lab[i]))
	end
	pgfsave("output/ir.pdf", axis)
	pgfsave("output/tikz/ir.tikz", axis)
end


function tikz_plot_ir_group(indiv_ir_price, cycle_length, dev_gains, set, show_cuts = true)	
	expected_nash_price = mean(nash_price, dims = 2)
	expected_coop_price = mean(coop_price, dims = 2)
	col = ["blue", "red", "green!90!lightgray", "orange"]
	dev_agent = 2
	dev_t = 1

	# save information structure cuts (works only if agents have the same prices)
	cuts = zeros(Int64,length(c_part))
	cuts[1] = c_part[1]
	for i in 2:length(c_part)
		cuts[i] = cuts[i-1] + c_part[i]
	end
	float_cut = prices[cuts,1] .+ (prices[2,1] - prices[1,1])/2

	axs = []
	for z in set
		shape = ["square" for i in 1:n_agents]
		shape[dev_agent] = "triangle"
		markersize = ["2.5pt" for i in 1:n_agents] 
		markersize[dev_agent] = "3pt"
		ir_len = size(indiv_ir_price[z][dev_t,dev_agent],1)
		dev_gains_ = round(dev_gains[z][dev_t,dev_agent,dev_agent], digits = 5)

		ax = @pgf Axis(
		    {
		        xlabel = raw"", ylabel = raw"", 
		        y_label_style = "{rotate=-90}",
				title = "",#string("Individual Impulse Response (devagent = ",dev_agent,", devt = ",dev_t,")"),
				xtick = [cycle_length[z]*ir_ext, cycle_length[z]*ir_ext+dev_t],			
				xticklabels = ["0", string(dev_t)],			
				yticklabels = [],
	       		ymin = minimum(prices)-0.02, ymax = maximum(prices)+0.02,
		       	xmin = 0.8, xmax = string(ir_len,".2"),
		        height = "7cm", width = "12cm",
		    },
		    HLine({ dashed, blue!50!darkgray }, nash_price[1]),
		    HLine({ dashed, blue!50!darkgray }, coop_price[1]),
		    HLine({ dashed, red!50!darkgray }, nash_price[2]),
		    HLine({ dashed, red!50!darkgray }, coop_price[2]),
		    VLine({ dashed, gray }, cycle_length[z]*ir_ext+dev_t),
			[raw"\node [draw,above left] at (current bounding box.south east) {",dev_gains_,"};"]

		);

		if length(c_part) < n_prices
			@pgf for i in keys(float_cut) 
				show_cuts == true && push!(ax, HLine({ dashed, green!30!lightgray, thick }, float_cut[i]))
			end
		end

		@pgf for i in 1:n_agents
		    pl = Plot(
		    	{
		            color = col[i],
		            mark  = shape[i], 
					mark_size = markersize[i],
					thick
		        },
		        Table(
		            x = 1:ir_len,
		            y = indiv_ir_price[z][dev_t,dev_agent][:,i]
		        )
		    ) 	
		    push!(ax, pl)
		end

		push!(axs, ax)
	end

	group_ax = @pgf GroupPlot(
    { group_style = { group_size="3 by 3", raw"vertical sep=15pt", raw"horizontal sep = 15pt" },
		xlabel=raw"$x$",
    }, axs...)

	pgfsave("output/ir_group.pdf", group_ax)
	pgfsave("output/tikz/ir_group.tikz", group_ax)
end


function tikz_plot_aggr_pg(last_epoch, epoch_profits)
	max_last_epoch = trunc(Int, mean(last_epoch) + std(last_epoch))
	profit_gains_ext = Array{Float64,3}(undef, max_last_epoch, n_agents, n_sessions)
	for z in 1:n_sessions
		for t in 1:max_last_epoch
			profit_gains_ext[t,:,z] = (epoch_profits[t,:,z]/n_episodes .- nash_profit) ./ (coop_profit - nash_profit)
			if t >= last_epoch[z]
				profit_gains_ext[t,:,z] = profit_gains_ext[last_epoch[z],:,z]
			end
		end
	end
	mean_profit_gains = mean(profit_gains_ext[1:max_last_epoch,:,:], dims = 3)[:,:,1]
	pg_min = minimum(mean_profit_gains)
	pg_max = maximum(mean_profit_gains)
	
	#eps = [1-(1-unique(eps0)[i].*unique(beta)[i]^t)^n_agents for t in 0:max_last_epoch-1, i in 1:length(unique(beta))]
	col = ["blue", "red", "green!90!lightgray", "orange"]

	axis = @pgf Axis(
	    {
	        xlabel = raw"$epoch$", ylabel = raw"$\Delta_t$", 
	        y_label_style = "{rotate=-90}",
			title = "",
	       	ymin = min(-0.1,pg_min), ymax = max(1.1,pg_max),
	       	xmin = 0, xmax = max_last_epoch+max_last_epoch/30,
			legend_pos="north west",
	    },
	    #Plot({color = "green!40!lightgray", opacity=0.3, thick, raw"forget plot"},Table(x = 1:max_last_epoch, y = eps,)),
	);
	@pgf for i in 1:n_agents
		pl = Plot(
			{
				color = col[i],
				no_markers,
				thick,
			},
			Table(x = 1:max_last_epoch, y = mean_profit_gains[:,i]))
		push!(axis, pl)
		push!(axis, LegendEntry("Agent "*string(i)))
	end
	pgfsave("output/pg.pdf",axis)
	pgfsave("output/tikz/pg.tikz",axis)
end


function tikz_plot_pg_group(last_epoch, epoch_profits, set)
	axs = []
	for z in set
		profit_gains = (epoch_profits[1:last_epoch[z],:,z]/n_episodes .- nash_profit') ./ (coop_profit' - nash_profit')
		pg_min = minimum(profit_gains)
		pg_max = maximum(profit_gains)
		col = ["blue", "red", "green!90!lightgray", "orange"]

		ax = @pgf Axis(
		    {
		        xlabel = raw"", ylabel = raw"", 
		        y_label_style = "{rotate=-90}",
				title = "",
				xticklabels = [], yticklabels = [],			
		       	ymin = min(-0.1), ymax = max(1.1),
		       	xmin = 0, xmax = last_epoch[z]+last_epoch[z]/30,
				height = "7cm", width = "12cm",
				legend_pos="north west",
		    },
			HLine({ dashed, gray }, 1),
			HLine({ dashed, gray }, 0),
		);
		@pgf for i in 1:n_agents
			pl = Plot(
				{
					color = col[i],
					no_markers,
					thick,
				},
				Table(x = 1:last_epoch[z], y = profit_gains[:,i]))
			push!(ax, pl)
			#push!(axis, LegendEntry("Agent "*string(i)))
		end
		push!(axs, ax)
	end
	group_ax = @pgf GroupPlot(
    { group_style = { group_size="3 by 3", raw"vertical sep=15pt", raw"horizontal sep = 15pt" },
		xlabel=raw"$x$",
    }, axs...)
	pgfsave("output/pg_group.pdf",group_ax)
	pgfsave("output/tikz/pg_group.tikz",group_ax)
end


in_session = @isdefined n_agents
if in_session == false
	const config, nash_price, coop_price, nash_profit, coop_profit, prices = load("output/data.jld", 
		"config", "nash_price", "coop_price", "nash_profit", "coop_profit", "prices")
	const beta = config["beta"]
	const eps0 = config["eps0"]
	const n_agents = length(beta)
	const n_prices = config["n_prices"]
	const p_nash = config["p_nash"]
	const p_coop = n_prices .- p_nash .+ 1
	const c_part = config["c_part"]
	const n_sessions = config["n_sessions"]
	const max_epochs = config["max_epochs"]
	const n_episodes = config["n_episodes"]
	const ir_ext = config["ir_ext"]
	const memory_length = config["memory_length"]
	const n_states = n_prices^(n_agents*memory_length)
	generate_tikz_plots()
end

