# A Morgan's code for Antarctic Ice Shelf benchmark dataset

### Dataset provided by C Baumhoer, DLR

### File structure:
Dataset:

ICE-BENCH

------ Envisat

------------ scenes

------------ masks

------------ test_envisat

------ ERS

------------ scenes

------------ masks

------------ test_ERS

------ Sentinel-1

------------ scenes

------------ masks

------------ test_s1


### Using latex on jasmin
Install script install-texlive.sh

Run: bash install-texlive.sh

Then:
export PATH="$HOME/texlive/$(date +%Y)/bin/$(ls $HOME/texlive/$(date +%Y)/bin | head -n1):$PATH"

