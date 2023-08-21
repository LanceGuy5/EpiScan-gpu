# Notes and commands for running GPU jobs on the Respublica HPC cluster

### Helpful Links
[CHOP-HPC-WIKI](https://wiki.chop.edu/display/RISUD/Respublica)

[slurm](https://slurm.schedmd.com)

 
1) `ssh` onto Respublica using your username-hpc9.research.chop.edu

For example,`ssh stearb-hpc9.research.chop.edu` 

If you don't have a VDI login node you need to first go to
[CHOP-VDI-LOGIN](https://wiki.chop.edu/pages/viewpage.action?pageId=261754446) and make one before doing ssh.


2) Load [slurm](https://slurm.schedmd.com) with `module load slurm`

3)  Either,
   a) Use the interactive `srun` command to have the output of your script print straight to the terminal
      instead of writing to a file like `sbatch` does. Use `srun` to debug or test script but dont run full analysis there.
      (Run script with `srun -p gpuq --gres=gpu:a100:2 --pty bash` )
    b) Use `sbatch` to 


-------------------------------

NOTES on submitting jobs using slurm
- 
- must do `chmod +x test.sh` before running
- `squeue --me`       # see my jobs

  
**Examples of Parameters**:
--mem=8G
-t  00:30:00 # time to run
-c 2 # number of CPUs

You can specify parameters as comments in the script like this:
#SBATCH
#SBATCH
#SBATCH


Also look into using GPU native Python packages like [cuPy](https://cupy.dev) or [cuDF](https://docs.rapids.ai/api/cudf/stable/).

