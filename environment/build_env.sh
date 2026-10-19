module load apptainer
# Build the SIF image to the exact path used by the makefile and jobs/test_model.sh
# (~/CONTAINERS/paecter, no trailing slash). apptainer build (without --sandbox)
# produces a single-file SIF image.
mkdir -p ~/CONTAINERS
apptainer build --force ~/CONTAINERS/paecter Singularity.paecter