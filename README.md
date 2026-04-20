# submission_76
Repository containing code and data for submission #76 for BuildSys '26. <br>
<img width="1387" height="499" alt="aleth_diagram" src="https://github.com/user-attachments/assets/20b58675-9967-4967-b8da-6f3c7c720944" /> <br>
Each of the components described in the architecture can be found under [https://github.com/buildsys76/submission_76/tree/main/code/experiments_figures_tables/results/aleth](https://github.com/buildsys76/submission_76/tree/main/code/experiments_figures_tables/results/aleth) alongside instructions for reproducibility.

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

## Further examples of data and simulation results
The raw prompt used to ask `aleth` for data is printed underneath each of the visualization titles. <br>
#### `Temperature` examples: <br>
<img width="300" height="300" alt="1_telemetry_plot" src="https://github.com/user-attachments/assets/6b1d8866-e7ce-4017-a504-8727755ad9e7" /> <br>
#### `CO2` examples: <br>
<img width="300" height="300" alt="1_office_indoor_co2telemetry_plot" src="https://github.com/user-attachments/assets/bbe09243-86b4-44ff-a7e2-46273bf0bb4f" /> <br>
#### `Occupancy` examples: <br>
<img width="300" height="300" alt="1_occupancytelemetry_plot" src="https://github.com/user-attachments/assets/3b6303c2-c0c2-4c19-a26a-33a984b391c2" /> <br>
#### `Electricity` examples (also with throughput): <br>
3-year data (100k sqft school) <img width="300" height="300" alt="telemetry_daily_mean_electricity_5" src="https://github.com/user-attachments/assets/05816409-e36d-4346-85a3-b4bd0e7cd1ce" /><br>
Latency per request for a 3-year electricity timeseries (100k sqft school) <img width="300" height="300" alt="llm_request_times_electricity_5" src="https://github.com/user-attachments/assets/4c212a59-8f70-4d5b-9bd6-0c61ceb5e7ad" /><br>
Latency per request for a 1-year electricity timeseries (100k sqft school) 
<img width="300" height="300" alt="llm_request_times_electricity_4" src="https://github.com/user-attachments/assets/b394d1b6-96b7-44c2-ab1c-d5d4a566b766" /><br>
#### `Energy metering` examples (school, 1k sqft): <br>
<img width="300" height="300" alt="telemetry_timeseries_5" src="https://github.com/user-attachments/assets/f8232b67-4053-46fe-9f95-19ad7faee407" />
<img width="300" height="300" alt="telemetry_timeseries_4" src="https://github.com/user-attachments/assets/e2aa1770-60aa-4bb3-b01c-1a0a73ecccc4" />
<img width="300" height="300" alt="telemetry_timeseries_3" src="https://github.com/user-attachments/assets/764003dd-0847-4148-916c-f43b173e6c4c" />
<img width="300" height="300" alt="telemetry_timeseries_2" src="https://github.com/user-attachments/assets/ec8c1489-34a8-4de9-a5f8-4afa3f3cd156" />
<img width="300" height="300" alt="telemetry_timeseries_1" src="https://github.com/user-attachments/assets/b468df3f-6eb8-46cb-8ffc-0ba3394a9ccd" />


#### `PM10` examples: <br>
<img width="300" height="300" alt="rd11_telemetry_plot" src="https://github.com/user-attachments/assets/e8a94f46-7827-442c-a56f-4a9d783f6cce" /> <br>
#### `Barrometric pressure` examples: <br>
<img width="300" height="300" alt="rd8_telemetry_plot" src="https://github.com/user-attachments/assets/ff1f4a76-62c3-48c5-86e0-fcfea426a527" /> <br>
#### `Humidity` examples: <br>
<img width="300" height="300" alt="rd9_telemetry_plot" src="https://github.com/user-attachments/assets/9f654249-7b0b-40d4-a9d3-a11d2d2a49f7" /> <br>

## Example of generalization capabilities to more "exotic" modalities `Water conductivity` example with plausible values
<img width="300" height="300" alt="water_conductivity_aleth" src="https://github.com/user-attachments/assets/0c260f26-669a-42b2-9711-edf6e5fef8a9" /> <br>
Aleth correctly inferred the unit of measurement, i.e., `µS/cm` and predicted plausible ranges ([https://atlas-scientific.com/blog/water-conductivity-range/?srsltid=AfmBOopMWGKpRsKt2VfqfNWXtQU48EqNbpXLhvO-Yx2X2ry8H55nrAAB](https://atlas-scientific.com/blog/water-conductivity-range/?srsltid=AfmBOopMWGKpRsKt2VfqfNWXtQU48EqNbpXLhvO-Yx2X2ry8H55nrAAB)).


## Failure cases for simulation
We assess robustness by executing each request up to three times and analyzing variability in outputs. Failures arise from two main sources. (1) Semantic ambiguity. For certain modalities (e.g., occupancy), interpretations may vary. For instance, occupancy can be modeled as binary presence, count-based values, or probabilistic ranges. This ambiguity can lead to outputs that differ from user expectations. Addressing this requires either improved intent understanding by the model or introducing higher-level abstractions (e.g., ontology-like specifications) to constrain interpretations. (2) Model and system limitations. Less capable models exhibit higher failure rates in structured outputs (e.g., malformed JSON), and backend limitations (e.g., concurrency issues in ollama) can introduce instability. These issues are largely orthogonal to aleth and are expected to improve with more robust inference backends and model capabilities. <br>
Examples of failure cases, where the ranges generated are either constant or resembling a rectangular function (https://en.wikipedia.org/wiki/Rectangular_function) which is again unlikely to be the case for real-world sensing measurements. <br>
<img width="300" height="300" alt="rd13_telemetry_plot" src="https://github.com/user-attachments/assets/42884d1d-854e-4242-b259-de34b61d5f2e" /> <br>
<img width="300" height="300" alt="error_occupancy_telemetry_plot" src="https://github.com/user-attachments/assets/4b0598a7-a1dc-4f64-9192-1f2bdc192dab" /> <br>
<img width="300" height="300" alt="rd6_telemetry_plot" src="https://github.com/user-attachments/assets/5082cb9e-d355-44e0-8ffb-2cf3f2fe1d8c" /> <br>


## Comparison against simulation workflows
Currently, as a comparison between alternatives for obtaining building sensor data, we consider both physics-based simulation workflows (e.g., EnergyPlus and tools built on top of it such as synconn_build) as a benchmark for required inputs, setup complexity, and achievable realism. These are contrasted with low-barrier approaches enabled by LLMs, which generate simulation code or data pipelines directly from natural language descriptions.

| Feature                            | Aleth                                                                     | EnergyPlus                      | synconn_build                                                                                  | AI Assistants (e.g., ChatGPT-like)                    |
| ---------------------------------- | ------------------------------------------------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Building model eequired            | No                                                                        | Yes                             | Yes                                                                                            |                No                                     |
| Natural language interface         | Yes                                                                       | No                              | No                                                                                             |                Yes                                    |
| Realism | Good for lightweight prototyping (plausible, but not fully physics-based) | Physics-based simulation | Physics-based (stochastic synthetic dataset generation for controls/ML benchmarking) | Unreliable (unrealistic synthesis over long duration) |

