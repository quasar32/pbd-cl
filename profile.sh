for ((i = 4096; i <= 262144; i += 4096))
do
  ./pbd -g $i -e
done
