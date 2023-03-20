# Energy cost of scientific software

In this exercise we will look at the profiling of a package I have written in Python, that performs data 
preprocessing and principal component analysis of monsoon data. The profiling is done with the help of a 
open-source tool, Scalene, a high-precision CPU and memory profiler.

## Run locally

- Clone this repository.

- Set up your conda environment with:

        $ conda env create -f environment.yml  # `mamba` works too for this command
        $ conda activate mssa-dev

- Launch Jupyter Lab with command `jupyter lab` and run `code-profiling.ipynb`.
