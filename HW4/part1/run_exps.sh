declare -a files=("pi_block_linear" "pi_block_tree" "pi_nonblock_linear" "pi_gather" "pi_reduce" )

for file in "${files[@]}"; do
  mpicxx "${file}.cc" -DSTORE_TIME -o "${file}"
  echo "compile ${file}.cc done."
done

parallel-scp -h hosts -r ~/HW4 ~

if [ -f "elapsed.csv" ]; then
  rm "elapsed.csv"
fi

for file in "${files[@]}"; do
  echo "run exp of ${file}..."
  for num_thread in 2 4 8 12 16; do
    for _ in {1..100}; do
      mpirun -np ${num_thread} --hostfile hosts "${file}" 1000000000
    done
  done
done
