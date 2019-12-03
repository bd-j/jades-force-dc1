#!/bin/bash

ngauss=3

band="F090W F115W F150W F200W F277W F335M F356W F410M F444W"
for B in $band; do
 for name in /Users/bjohnson/Projects/jades_force/data/psf/*${B}*fits; do
   echo $name
   b="$(tr [A-Z] [a-z] <<< "$B")"
   echo $b
   python make_psf_mixture.py --band=os2_jwst_${b} --path_psf=$name --oversample=2 --ngauss=$ngauss
  done
done
