#!/bin/bash
#
# ESBMC - Benchmark Witnesses Validation
#
#               Universidade Federal do Amazonas - UFAM
# Author:       Hussama Ismail <hussamaismail@gmail.com>
#
# ------------------------------------------------------
#
# Check if the graphml generated is correct using a 
# configurated validation tool as (CPAChecker, CBMC)
# 
# 
# Usage Example:
# $ sh script source_file.c graphml_file
#
# ------------------------------------------------------

# VERIFICATION WITNESSES PARAMETERS 
WITNESSES_VALIDATOR_FOLDER="./files/cpachecker/";
WITNESSES_VALIDATOR_EXECUTABLE="scripts/cpa.sh";
WITNESSES_VALIDATOR_PARAMETERS="-64 -explicitAnalysis-NoRefiner -setprop analysis.checkCounterexamples=false -setprop cpa.value.merge=SEP -setprop cfa.useMultiEdges=false -setprop parser.transformTokensToLines=true";
WITNESSES_VALIDATOR_PROPERTY="PropertyERROR.prp";
WITNESSES_VALIDATOR_MEMSAFETY_PARAMETERS="-64 -preprocess -sv-comp14--memorysafety -spec config/specification/cpalien-leaks.spc -spec config/specification/TerminatingFunctions.spc -setprop cfa.useMultiEdges=false -setprop parser.transformTokensToLines=true";

SOURCE_CODE=$1
GRAPHML=$2

IS_MEMORY_SAFETY=$(echo $3 | grep -i "memory-safety" | wc -l);
IS_FAILED_RESULT_EXPECTED=$(echo $SOURCE_CODE | egrep -i "unsafe|false" | wc -l); 

if ([ -z ${SOURCE_CODE} ] || [ -z ${GRAPHML} ]); then
   echo "Is necessary specify a source code file and a graphml. (use: $0 \"c_file\" \"graphml\")";
   exit 0;
fi

# CHECK IF GRAPHML EXISTS
if ([ $IS_FAILED_RESULT_EXPECTED -eq 0 ] && [ ! -s $GRAPHML ]); then
   echo "correct";
   exit 0;
elif ([ $IS_FAILED_RESULT_EXPECTED -eq 1 ] && [ ! -s $GRAPHML ]); then
   echo "incorrect";
   exit 0;
fi

# CALL WITNESSES VALIDATOR
cd $WITNESSES_VALIDATOR_FOLDER;

if [ $IS_MEMORY_SAFETY -eq 1 ]; then
   OUTPUT=$($WITNESSES_VALIDATOR_EXECUTABLE $WITNESSES_VALIDATOR_MEMSAFETY_PARAMETERS -spec $(echo $GRAPHML) $(echo $SOURCE_CODE) 2> /dev/null)
else
   OUTPUT=$($WITNESSES_VALIDATOR_EXECUTABLE $WITNESSES_VALIDATOR_PARAMETERS -spec $(echo $GRAPHML) -spec $(echo $WITNESSES_VALIDATOR_PROPERTY) $(echo $SOURCE_CODE) 2> /dev/null)
fi

# CHECK WITNESSES VALIDATOR RESPONSE
IS_FAILED_RESULT_GRAPHML=$(echo "$OUTPUT" | grep -i "Verification result:" | cut -d ":" -f2 | cut -d "." -f1 | cut -d " " -f2 | grep -i "FALSE" | wc -l);

if ([ $IS_FAILED_RESULT_EXPECTED -eq 1 ] && [ $IS_FAILED_RESULT_GRAPHML -eq 1 ]); then
   echo "correct";
else
   echo "incorrect";
fi
