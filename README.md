<div style="background-color:#ff4d4f; color:white; padding:12px; border-radius:6px; text-align:center; font-weight:bold;">
We are currently updating the repository to address the reviewers comments
</div>

# submission_76
Repository containing code and data for submission #76 for BuildSys '26.

## Code for experiments and figures
Code used in the experiments can be found under `code`. For reproducibility in particular, check: `code/experiments_figures_tables/README.md`.

## Data
Data used in the experiments can be found under `data`.

## Discussion and examples
### Off-the shelf LLMs
While capable to simulate short-horizon telemetry (even with more sophisticated regimes, e.g., with anomalies), off-the-shelf cannot generate long-horizon data. Typically, they output either a proposed solution in code rather than sensor data, or they truncate output. We append some examples below (these can be found extensively under `code/experiments_figures_tables/results/off_the_shelf_llms/quantitative` - each `.txt` file indicated the amount of days that were requested, e.g., `gpt-oss_20b_003_days.txt`, which means data for three days was requested). <tab>

For example, we append below a snapshot from an actual LLM response (`submission_76/code/experiments_figures_tables/results/off_the_shelf_llms/quantitative/quantitative_results_incremental_by_no_days/round_10_llama3.1_8b_365_days.txt`) of how output is truncated (for a prompt that requests data for 365 days)
```bash
Here is the generated electrical metering data for a 10,000 sq ft building from January 1st, 2016 to December 31st, 2016:

01/01/16 00:00,0.001
01/01/16 01:00,0.002
01/01/16 02:00,0.003
01/01/16 03:00,0.004
...
( output truncated for brevity )

12/31/16 23:00,1008.923

However, since the output is too large to be displayed in its entirety here, I'll provide a script that generates this data and you can run it yourself.

Here's a Python script using numpy and pandas libraries to generate the data:

```python
import numpy as np
import pandas as pd
[...]
```
Alternatively, models can also simply refuse to generate the output (based on the volume of data requested, i.e., 365 days in this example taken from `submission_76/code/experiments_figures_tables/results/off_the_shelf_llms/quantitative/quantitative_results_incremental_by_no_days/round_10_gpt-oss_20b_365_days.txt`)
```bash
I’m sorry, but I can’t provide that.
```
