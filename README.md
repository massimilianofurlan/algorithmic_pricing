# Algorithmic pricing 
Replication of [Calvano et al. (2020)](https://www.aeaweb.org/articles?id=10.1257/aer.20190623).


### Dependencies (Julia): 
```
import Pkg; 
Pkg.add("TOML"); 
Pkg.add("Statistics"); 
Pkg.add("ProgressMeter"); 
Pkg.add("Random"); 
Pkg.add("JLD"); 
Pkg.add("Plots");
Pkg.add("PGFPlotsX");
Pkg.add("PrettyTables");
```

### Usage:

Use config.TOML to change the initial configurations of the experiment. Run ``` julia --threads=auto main.jl``` in terminal.
