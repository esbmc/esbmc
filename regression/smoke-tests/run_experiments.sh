#!/bin/bash

CBMC=../bin/esbmc
#Z3 is the default solver
SOLVER= 
BINARY=
CLAIM=--claim
OVERFLOWCHECK=--overflow-check
TIMEOUT=3600 #1h=3600s
RM=rm
filename=*.c
claim=Claim
VC_total=0
VC_successful=0
VC_violated=0
VC_failed=0
total_decision_time=0
time=0
encoding_time=0
total_encoding_time=0
encoding_VC_total=0
decision_time=0
results_file=results.csv
number_of_files=0

#list of files
SOURCES=$(ls *.c)
#SOURCES=$(ls | grep -v "\.")

#set timeout
ulimit -t $TIMEOUT

print_results()
{
  echo " "
  echo "--------------------------------------------------"
  echo "     Experimental Results for" $file
  echo "--------------------------------------------------"
  echo "Lines of Code:" $loc
  echo "Bound:" $bound
  echo "Total Claim(s):"  $VC_total
  echo "Passed Claim(s):" $VC_successful
  echo "Violated Claim(s):" $VC_violated
  echo "Failed Claim(s):" $VC_failed
  echo "Total Time:" $total_time "s"
  echo "Encoding Time:" $total_encoding_time "s"
  echo "Decision Procedure Time:" $total_decision_time "s"
  echo "--------------------------------------------------"
  echo "##################################################"
  echo " "
}

write2file()
{
  echo $file ";" $loc ";" $bound ";" $VC_total ";" $total_encoding_time ";" $total_decision_time ";" $total_time ";" $VC_successful ";" $VC_violated ";" $VC_failed >> $results_file
}

init_vars()
{
  loc=0
  bound=0
  VC_total=0
  VC_successful=0
  VC_violated=0
  VC_failed=0
  total_time=0
  total_decision_time=0
  decision_time=0
  encoding_time=0
  encoding_VC_total=0
  total_encoding_time=0
}

check_encoding_time()
{
  START=$(date +%s)
  if [ ! -f $file.bound ];
  then
    $CBMC $SOLVER $OVERFLOWCHECK $BINARY $file > $file.tmp
  else 
    $CBMC $SOLVER $OVERFLOWCHECK --unwind $bound $BINARY $file > $file.tmp
  fi
  sleep 1
  END=$(date +%s)
  time=$(( $END - $START ))

  #check for time-out
  if [ $time -ge $TIMEOUT ];
  then
    total_encoding_time=0
  else
	#the command date +%s does not produce milliseconds
	if [ $time -eq 0 ];
	  then
	    time=1
	  fi

	decision=`cat $file.tmp | grep 'Runtime decision procedure: '`

    decision_tmp=${decision##R*:} 
	decision_time=${decision_tmp%s*}

	if [ -n "$decision_time" ]; 
	then
      total_encoding_time=`echo $time - $decision_time | bc -l`
	else
      total_encoding_time=$time
	fi

	total_time_cnstr_prop=$time
	decisition_time_cnstr_prop=$decision_time
  fi
}

list_file()
{
  echo "SOURCES:" $SOURCES

  #write header of the CSV file
  echo "Module; #L; B; #P; Encoding Time; Decision Procedure Time; Total Time; Passed; Violated; Failed" > $results_file

  echo "##################################################"
  echo "List of ANSI-C programs:"
  echo " "
  for file in $SOURCES
  do
    echo "$file"
    number_of_files=`expr $number_of_files + 1`
  done
  echo "Total number of ANSI-C programs:" $number_of_files
}

list_file

for file in $SOURCES
do
echo " "
echo "##################################################"

  if [ ! -f $file.bound ];
  then
    echo "Running program" $file
  else 
    bound=`cat $file.bound | grep ^[0-9]`
    echo "Running program" $file "with bound" $bound
  fi

  matches=`$CBMC $OVERFLOWCHECK --show-claims $BINARY $file | grep "$claim"`

  for claim_nr in $matches
  do
	#store number of claims
    VC_total=`expr $VC_total + 1`
  done

  VC_total=`expr $VC_total / 2`

  i=1
  while [ $i -le $VC_total ] 
  do

	echo " "
	echo "Checking property" $i "of" $VC_total	
	
	START=$(date +%s)
    if [ ! -f $file.bound ];
    then
      $CBMC $SOLVER $OVERFLOWCHECK $CLAIM $i $BINARY $file > $file.tmp
    else 
	  $CBMC $SOLVER $OVERFLOWCHECK $CLAIM $i --unwind $bound $BINARY $file > $file.tmp
    fi
	END=$(date +%s)
	time=$(( $END - $START ))

	#check for time-out
	if [ $time -ge $TIMEOUT ];
	then
	  decision_time=$time
      VC_failed=`expr $VC_failed + 1`
	else
	  #the command date +%s does not produce milliseconds
	  if [ $time -eq 0 ];
	  then
	    time=1
	  fi


	  result=`cat $file.tmp | grep 'SUCCESSFUL'`
	  decision=`cat $file.tmp | grep 'Runtime decision procedure: '`
	  unwinding=`cat $file.tmp | grep 'unwinding' | cut -c 3-`

	  $RM -f *.tmp

      decision_tmp=${decision##R*:} 
	  decision_time=${decision_tmp%s*}

	  if [ -n "$decision_time" ]; 
	  then
        encoding_time=`echo $time - $decision_time | bc -l`
	  else
        decision_time=0
        encoding_time=$time
	  fi

      #cat dp_time.tmp | grep : | cut -c 29- | grep "s"

	  if [ -n "$result" ]; 
	  then
	    VC_successful=`expr $VC_successful + 1`
	    echo "Status: Passed"	  
	  else
		if [ -n "$unwinding" ];
		then
	      VC_successful=`expr $VC_successful + 1`
		  VC_violated=$VC_violated
		  echo "Status: unwinding assertion loop"
		else
	      VC_violated=`expr $VC_violated + 1`
	      echo "Status: Violated"
		fi
	  fi


	  encoding_VC_total=`expr $encoding_VC_total + 1`
	  echo "Encoding Time:" $encoding_time "s"
	fi

	  ##store total time to check all properties
	  total_decision_time=`echo $total_decision_time + $decision_time | bc -l`

	  echo "Decision Procedure Time:" $decision_time "s"
      echo "Verification Time:" $time "s"

    i=`expr $i + 1`
  done


  ##calculate encoding time
  echo " "
  echo "Computing time to check all constraints and properties... "

  check_encoding_time

  loc=`cat $file.c | wc -l`
  total_time=`echo $total_decision_time + $total_encoding_time | bc -l`

  if [ $VC_failed -eq 0 ];
  then
	total_time=$total_time_cnstr_prop
    total_decision_time=$decisition_time_cnstr_prop
  fi

  print_results
  write2file
  init_vars
 
done


