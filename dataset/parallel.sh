for c in lits dlds kits amos brats bhsd; do
  nohup python3 -u generate_modorgan.py --only $c > gen6_$c.log 2>&1 &
done
