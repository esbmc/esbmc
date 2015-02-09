#!/bin/bash
#
# ESBMC - Benchmark Runner
#
#               Universidade Federal do Amazonas - UFAM
# Author:       Hussama Ismail <hussamaismail@gmail.com>
#
# ------------------------------------------------------
#
# Script that generate a benchmark report using competition style.
# 
# Usage Example:
# $ sh script folder/testcase
#
# ------------------------------------------------------
#
# History:
#
#  v0.1 2014-04-14, Hussama Ismail: 
#     - Initial Version
#  v0.2 2014-04-25, Hussama Ismail:
#     - Adding Requirements 
#  v0.3 2014-04-29, Hussama Ismail:
#     - Adjust html to order by columns
#  v0.4 2014-08-21, Hussama Ismail:
#     - Modifing the source method to get .i, and .c that 
#       hasn't a .i file.
#  v0.5 2014-10-14, Hussama Ismail:
#     - Adding witnesses verification support and
#       updating competition point values
#  v0.6 2015-01-19, Hussama Ismail:
#     - Updating with support to -c param for competition
#       (use -c prop.prp). It's send to esbmc-wrapper-script.sh
#  v0.7 2015-01-20, Hussama Ismail:
#     - Include a optional parameter (--no-witness) to ignore witnesses
#       verification

# DEPENDENCY PARAMETERS
ESBMC_WRAPPER_SCRIPT="./files/scripts/esbmc-wrapper-script.sh";
OUTPUT_REPORT_FILE="report-output-recursion-plain.html";
SCRIPT_VERSION="0.7";

# VERIFICATION WITNESSES
ESBMC_VERIFICATION_WITNESSES_SCRIPT="./files/scripts/esbmc-verification-witnesses-script.sh";

SOURCE_PARAMETERS=$@;
# CHECK PARAMETERS
if [ ${#@} -eq 0 ]; then
    echo "Is necessary specify a benchmark or a folder. (use: $0 benchmarks)";
    exit 0;
fi

# CHECK IF IS WITNESS
IS_NOT_WITNESS=$( echo $SOURCE_PARAMETERS | grep "no-witness" | wc -l );
if [ $IS_NOT_WITNESS -eq 1 ]; then
   SOURCE_PARAMETERS=$(echo ${SOURCE_PARAMETERS[@]/--no-witness});
fi

# SET -c PARAMETER (OPTIONAL)
while getopts "c:h" arg; do
    case $arg in
       	c)            
	    ESBMC_WRAPPER_SCRIPT="$ESBMC_WRAPPER_SCRIPT -c $OPTARG";
	    PARAMS=$SOURCE_PARAMETERS;
            PARAM_REMOVE_C=$(echo ${PARAMS[@]/-c})
            PARAM_REMOVE_C_PARAM=$(echo ${PARAM_REMOVE_C[@]/$OPTARG})                     
            SOURCE_PARAMETERS=$PARAM_REMOVE_C_PARAM;
            ;;
    esac
done

SOURCES="";
for current_source in "$SOURCE_PARAMETERS"; do
    FILES_I=$(find $current_source  -type f \( -iname "*.i" \));
    FILES_C=$(find $current_source -type f \( -iname "*.c" \))
    SOURCES=$(echo $SOURCES $FILES_I);  
    for c_file in ${FILES_C[@]}; do
	# check if exist a same .i file
        c_filename_lenght=${#c_file};
	c_filename_without_extension=${c_file:0:$((c_filename_lenght - 2))};	       
       	already_exists_i_file=$(echo ${FILES_I[@]} | grep -i ${c_filename_without_extension} | wc -l);	
	if [ $already_exists_i_file -eq 0 ]; then
	   SOURCES=$(echo $SOURCES $c_file);
        fi
    done
done

QTD_I_FILES=$(echo $SOURCES | grep -o "\.i$" | wc -l);
QTD_C_FILES=$(echo $SOURCES | grep -o "\.c$" | wc -l);
QTD_FILES=$((QTD_I_FILES + QTD_C_FILES));
if [ $QTD_FILES -eq 1 ]; then
   IS_SINGLE_FILE=1;
else 
   IS_SINGLE_FILE=0;
fi

# ESBMC PARAMS
ESBMC_EXECUTABLE=$(cat $ESBMC_WRAPPER_SCRIPT | grep "path_to_esbmc" | head -n1 | cut -d"=" -f2)
ESBMC_VERSION="ESBMC $($ESBMC_EXECUTABLE --version)"
ESBMC_PARAMS=$(cat $ESBMC_WRAPPER_SCRIPT | grep global_cmd_line | head -n1 | cut -d '"' -f2)

# SYSTEM INFO
DATE_EXECUTION=$(date)
INITIAL_TIMESTAMP=$(date +%s)
CPU_CORE_NUMBER=$(cat /proc/cpuinfo | grep processor | wc -l)
CPU_INFO="CPU:$(cat /proc/cpuinfo | grep "model name" | tail -n1 | cut -d ":" -f2)"
MEM_INFO="RAM: $(cat /proc/meminfo | grep "MemTotal" | cut -d ":" -f2 | cut -d " " -f8) kB"

# HTML CONTENT
HTML_TABLE_HEADER="<table style=\"width: 100%\"><thead><tr id=\"tool\"><td style=\"width: 60%\">Tool</td><td colspan=\"2\">$ESBMC_VERSION</td></tr><tr id=\"system\"><td>System</td><td colspan=\"2\">$CPU_INFO - $MEM_INFO</td></tr><tr id=\"date\"><td>Date of run</td><td colspan=\"2\">$DATE_EXECUTION</td></tr><tr id=\"options\"><td>Options</td><td colspan=\"2\">$ESBMC_PARAMS</td></tr></thead></table><table id=\"datatable\" class=\"tablesorter\" style=\"width: 100%; margin-top: 3px\"><thead><tr id=\"columnTitles\"><th style=\"width: 60%; text-align: left\" class=\"clickable\"><span style=\"font-size: x-small; font-weight: normal; text-align: left;\">$(echo $SOURCE_PARAMETERS | sed -e "s/ /<br>/g")</span></th><th style=\"width: 12%\" colspan=\"1\" class=\"clickable\">status</th><th colspan=\"1\" style=\"width: 16%\" class=\"clickable\">verification witnesses</th><th style=\"width: 12%\ colspan=\"1\" class=\"clickable\">time(s)</th><th style=\"display: none\">is Failed?</th></tr></thead><tbody>"

# REPORT CONTROL
TOTAL_FILES=0
TOTAL_UNKNOWN=0
TOTAL_ERROR=0
CORRECT_RESULTS=0
CORRECT_TRUES=0
CORRECT_FALSES=0
FALSE_POSITIVES=0
FALSE_NEGATIVES=0
MAX_SCORE=0
TOTAL_POINTS=0
TOTAL_WITNESSES=0
CORRECT_WITNESSES=0
INCORRECT_WITNESSES=0

cp ./files/report/header.html $OUTPUT_REPORT_FILE
echo $HTML_TABLE_HEADER >> $OUTPUT_REPORT_FILE

echo "*** ESBMC Benchmark Runner v$SCRIPT_VERSION ***"
echo "";
echo "Tool: $ESBMC_VERSION"
echo "Date of run: $(date)"
echo "System: $CPU_INFO $MEM_INFO"
echo "Options: $ESBMC_PARAMS"
echo "Source: $@"
echo "Total Tasks: $QTD_FILES"
echo "";

for file in $SOURCES; do
   TOTAL_FILES=$((TOTAL_FILES + 1));
   FILENAME=$file

   if [ $IS_SINGLE_FILE -eq 0 ]; then
      FILENAME=$(echo $file);
      for current_source in "$@" ; do
         ACC=$(echo $current_source | grep -o "\<c\>" | wc -l);
         ACI=$(echo $current_source | grep -o "\<i\>" | wc -l);
         AC=$((ACC+ACI));
         if [ $AC -ge 1 ]; then
            continue;
         fi
         TEMP=$(echo $file | sed -e "s:$current_source::")       
  	 if [ $(expr length $TEMP) -lt $(expr length $FILENAME) ]; then 
            FILENAME=$(echo $TEMP);
         fi
      done
   fi

   echo "RUNNING: " $FILENAME
   
   EXPECTED_FAILED_RESULT=$(echo $file | egrep -i "unsafe|false" | wc -l); 
   if [ $EXPECTED_FAILED_RESULT -eq 1 ]; then
      MAX_SCORE=$((MAX_SCORE + 1));
   else
      MAX_SCORE=$((MAX_SCORE + 2));
   fi   

   INITIAL_EXECUTION_TIMESTAMP=$(date +%s)
   OUT=$(sh $ESBMC_WRAPPER_SCRIPT $file;)
   FINAL_EXECUTION_TIMESTAMP=$(date +%s)

   ERROR=$(echo $OUT | grep "ERROR" | wc -l);
   FAILED=$(echo $OUT | grep "FALSE" | wc -l); 
   UNKNOWN=$(echo $OUT | grep "UNKNOWN" | wc -l);  
   TIME_OUT=$(echo $OUT | grep "TIMEOUT" | wc -l);  
   SUCCESS=$(echo $OUT | grep "TRUE" | wc -l);  
   INCORRECT_RESULT=0;  
   TIME=$((FINAL_EXECUTION_TIMESTAMP - INITIAL_EXECUTION_TIMESTAMP));
 
   CSS_CLASS="";
   RESULT_TEXT=""; 
   WITNESSES_CSS="";
   WITNESSES_TEXT="-";

   if [ $TIME_OUT -eq 1 ] || ([ $FAILED -eq 0 ] && [ $SUCCESS -eq 0 ] && [ $UNKNOWN -eq 0 ] && [ $ERROR -eq 0 ]); then
      CSS_CLASS="status error";
      RESULT_TEXT="timeout";
      INCORRECT_RESULT=1
      TOTAL_UNKNOWN=$((TOTAL_UNKNOWN + 1));
   elif [ $ERROR -eq 1 ] && ([ $FAILED -eq 0 ] && [ $SUCCESS -eq 0 ] && [ $UNKNOWN -eq 0 ] && [ $TIMEOUT -eq 0 ] ); then
      CSS_CLASS="wrongProperty";
      RESULT_TEXT="ERROR"; 
      INCORRECT_RESULT=1
      TOTAL_ERROR=$((TOTAL_ERROR + 1));
      echo $(echo -e "\033[1;35mERROR\033[0m" | cut -d " " -f2) "in $TIME""s";
   elif [ $UNKNOWN -eq 1 ] || ([ $FAILED -eq 0 ] && [ $SUCCESS -eq 0 ] && [ $TIME_OUT -eq 0 ] && [ $ERROR -eq 0 ] ); then
      CSS_CLASS="status unknown";
      RESULT_TEXT="unknown"; 
      echo $(echo -e "\033[0;33munknown\033[0m" | cut -d " " -f2) "in $TIME""s";
      INCORRECT_RESULT=1;
      TOTAL_UNKNOWN=$((TOTAL_UNKNOWN + 1));
   elif [ $EXPECTED_FAILED_RESULT -eq 1 ] && [ $FAILED -eq 1 ]; then
      CSS_CLASS="correctProperty";
      RESULT_TEXT="false(label)";
      CORRECT_RESULTS=$((CORRECT_RESULTS + 1));
      CORRECT_FALSES=$((CORRECT_FALSES + 1));
      ### VALIDATE WITNESSES ###
      if [ $IS_NOT_WITNESS -eq 0 ]; then 
         GRAPHML=$(echo $OUT | grep Counterexample | cut -d ":" -f2 | cut -d " " -f2)
	 WITNESSES_RESPONSE=$($ESBMC_VERIFICATION_WITNESSES_SCRIPT $file $GRAPHML);
	 TOTAL_WITNESSES=$((TOTAL_WITNESSES+1)); 
         IS_INCORRECT_WITNESSES=$(echo $WITNESSES_RESPONSE | grep "incorrect" | wc -l);
         if [ $IS_INCORRECT_WITNESSES -eq 1 ]; then
	    WITNESSES_CSS="wrongProperty";
            WITNESSES_TEXT="incorrect";
	    echo $(echo -e "\033[0;32mfalse(label)\033[0m" | cut -d " " -f2) "in $TIME""s ~ witnesses status: $(echo -e "\033[0;31mincorrect\033[0m" | cut -d " " -f2) ($GRAPHML)";
         else
            WITNESSES_CSS="correctProperty";
            WITNESSES_TEXT="correct";
	    CORRECT_WITNESSES=$((CORRECT_WITNESSES+1));
	    echo $(echo -e "\033[0;32mfalse(label)\033[0m" | cut -d " " -f2) "in $TIME""s ~ witnesses status: $(echo -e "\033[0;32mcorrect\033[0m" | cut -d " " -f2) ($GRAPHML)"
         fi
      else
        WITNESSES_CSS="";
        WITNESSES_TEXT="-";	
        echo $(echo -e "\033[0;32mfalse(label)\033[0m" | cut -d " " -f2) "in $TIME""s"
      fi
      ##########################
   elif [ $EXPECTED_FAILED_RESULT -eq 1 ] && [ $FAILED -eq 0 ]; then
      CSS_CLASS="wrongProperty";
      RESULT_TEXT="true";
      FALSE_POSITIVES=$((FALSE_POSITIVES + 1));
      echo $(echo -e "\033[0;31mtrue\033[0m" | cut -d " " -f2) "in $TIME""s";
      INCORRECT_RESULT=1;
   elif [ $EXPECTED_FAILED_RESULT -eq 0 ] && [ $FAILED -eq 1 ]; then
      CSS_CLASS="wrongProperty";
      RESULT_TEXT="false(label)";
      FALSE_NEGATIVES=$((FALSE_NEGATIVES + 1));
      echo $(echo -e "\033[0;31mfalse(label)\033[0m" | cut -d " " -f2) "in $TIME""s";
      INCORRECT_RESULT=1;
   elif [ $EXPECTED_FAILED_RESULT -eq 0 ] && [ $FAILED -eq 0 ]; then
      CSS_CLASS="correctProperty";
      RESULT_TEXT="true";
      CORRECT_RESULTS=$((CORRECT_RESULTS + 1));
      CORRECT_TRUES=$((CORRECT_TRUES + 1 ));
      echo $(echo -e "\033[0;32mtrue\033[0m" | cut -d " " -f2) "in $TIME""s";
   fi

   HTML_ENTRY="<tr><td>$FILENAME</td><td class=\"$CSS_CLASS\">$RESULT_TEXT</td><td class=\"$WITNESSES_CSS\" align=\"center\">$WITNESSES_TEXT</td><td class=\"unknownValue\">$TIME&nbsp;</td><td style=\"display: none\">$INCORRECT_RESULT</td></tr>" 
   echo $HTML_ENTRY >> $OUTPUT_REPORT_FILE
done
FINAL_TIMESTAMP=$(date +%s)

# CALCULATE POINTS
TOTAL_POINTS=$((TOTAL_POINTS + 2 * CORRECT_TRUES));
TOTAL_POINTS=$((TOTAL_POINTS + CORRECT_FALSES));
TOTAL_POINTS=$((TOTAL_POINTS - 12 * FALSE_POSITIVES));
TOTAL_POINTS=$((TOTAL_POINTS - 6 * FALSE_NEGATIVES));
TOTAL_EXECUTION_TIME=$((FINAL_TIMESTAMP - INITIAL_TIMESTAMP));

# HTML CONTENT
HTML_TABLE_FOOTER="</tbody></table><table style=\"width: 100%; margin-top: 2px\"><tfoot><tr><td style=\"width: 60%\">total files</td><td style=\"width: 12%\">$TOTAL_FILES</td><td style=\"width: 16%\">$TOTAL_WITNESSES</td><td class=\"unknownValue\" style=\"width: 12%\">$TOTAL_EXECUTION_TIME&nbsp;</td></tr><tr><td title=\"(no bug exists + result is SAFE) OR (bug exists + result is UNSAFE) OR (property is violated + violation is found)\">correct results</td><td>$CORRECT_RESULTS</td><td style=\"width: 16%\">$CORRECT_WITNESSES</td><td>-</td></tr><tr><td title=\"bug exists + result is SAFE\">false negatives</td><td style=\"width: 12%\">$FALSE_NEGATIVES</td><td style=\"width: 16%\"></td><td>-</td></tr><tr><td title=\"no bug exists + result is UNSAFE\">false positives</td><td style=\"width: 12%\">$FALSE_POSITIVES</td><td style=\"width: 16%\"></td><td>-</td></tr> <tr><td title=\"bug exists + result is SAFE\">total unknown</td><td style=\"width: 12%\">$TOTAL_UNKNOWN</td><td style=\"width: 16%\"></td><td style=\"width: 12%\">-</td></tr>  <tr><td title=\"17 safe files, 15 unsafe files\">score ($TOTAL_FILES files, max score: $MAX_SCORE)</td><td style=\"width: 12%\" class=\"score\">$TOTAL_POINTS</td><td class=\"score\"></td><td style=\"width: 16%\"></td></tr></tfoot></table></center><br><a href=\"#\" onclick=\"javascript:orderByCorrectIncorrectResults()\">order by correct / incorrect results</a></body></html>"

echo "";
echo "*** RESULTS *** ";
echo "Total Files: $TOTAL_FILES in $TOTAL_EXECUTION_TIME""s";
echo "Correct Results: $CORRECT_RESULTS";
echo "False Negatives: $FALSE_NEGATIVES";
echo "False Positives: $FALSE_POSITIVES";
echo "Total Unknown: $TOTAL_UNKNOWN";
echo "Score ($TOTAL_FILES files, max score: $MAX_SCORE): $TOTAL_POINTS";

echo $HTML_TABLE_FOOTER >> $OUTPUT_REPORT_FILE;
echo "";
echo "Report file generated: $OUTPUT_REPORT_FILE";
