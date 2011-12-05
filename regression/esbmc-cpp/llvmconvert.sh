SOURCES=$(ls -d */)
qtyok=0
qtyfail=0

for i in $SOURCES
do
	cd $i
	echo $i
	$(./convert.sh)
    if [ $? -eq 0 ]; then qtyok=$(expr $qtyok + 1);
    else qtyfail=$(expr $qtyfail + 1); fi

	cd ..
done
echo
echo $qtyok   "Passaram"
echo $qtyfail "Falharam"
echo $(expr $qtyok + $qtyfail)

