# Step count competition!
This repo contains code to process and visualise metrics derived from accelerometer data, such as step count.
The simplest version is a step count competition between two teams, team black and team teal, where the team with the highest median steps wins. However, you can come up with more nuanced comparisons.

# Installation.
To run `run.sh`, which processes the `black/` and `teal/` folders of accelerometer files (`.cwa`), you need to install [stepcount](https://github.com/OxWearables/stepcount/tree/main). I have found that some of the package versions needed for stepcount conflict with those used to run the `visualise.py` script, so I would recommend having separate conda environments to run stepcount and to run the visualisation script.

To run the visualise.py script, you need:
- numpy
- matplotlib
- pandas