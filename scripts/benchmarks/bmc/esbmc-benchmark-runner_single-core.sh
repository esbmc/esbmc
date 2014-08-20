#!/bin/bash
#
# ESBMC - Benchmark Runner (Single-Core)
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
#

# DEPENDENCY PARAMETERS
ESBMC_WRAPPER_SCRIPT="./esbmc_wrapper_script.sh";
OUTPUT_REPORT_FILE="report.html";
SCRIPT_VERSION="0.3";

# DIRECTORY
SOURCES="";
for current_source in "$@" ; do
    FILES=$(find $current_source -type f \( -iname "*.i" \))
    SOURCES=$(echo $SOURCES $FILES);
done
if [ $(echo $SOURCES | grep -o "\<i\>" | wc -l) -eq 1 ]; then
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
HTML_TABLE_HEADER="<table style=\"width: 100%\"><thead><tr id=\"tool\"><td style=\"width: 60%\">Tool</td><td colspan=\"2\">$ESBMC_VERSION</td></tr><tr id=\"limits\"><td>Limits</td><td colspan=\"2\"></td></tr><tr id=\"system\"><td>System</td><td colspan=\"2\">$CPU_INFO - $MEM_INFO</td></tr><tr id=\"date\"><td>Date of run</td><td colspan=\"2\">$DATE_EXECUTION</td></tr><tr id=\"options\"><td>Options</td><td colspan=\"2\">$ESBMC_PARAMS</td></tr></thead></table><table id=\"datatable\" class=\"tablesorter\" style=\"width: 100%; margin-top: 3px\"><thead><tr id=\"columnTitles\"><th style=\"width: 60%; text-align: left\" class=\"clickable\"><span style=\"font-size: x-small; font-weight: normal; text-align: left;\">$(echo $@ | sed -e "s/ /<br>/g")</span></th><th style=\"width: 20%\" colspan=\"1\" class=\"clickable\">status</th><th colspan=\"1\" class=\"clickable\">time(s)</th><th style=\"display: none\">is Failed?</th></tr></thead><tbody>"

# REPORT CONTROL
TOTAL_FILES=0
CORRECT_RESULTS=0
CORRECT_TRUES=0
CORRECT_FALSES=0
FALSE_POSITIVES=0
FALSE_NEGATIVES=0
MAX_SCORE=0
TOTAL_POINTS=0

cp ./REPORT/header.html $OUTPUT_REPORT_FILE
echo $HTML_TABLE_HEADER >> $OUTPUT_REPORT_FILE

echo "*** ESBMC Benchmark Runner (Single-Core) v$SCRIPT_VERSION ***\n"
echo "Tool: $ESBMC_VERSION"
echo "Date of run: $(date)"
echo "System: $CPU_INFO $MEM_INFO"
echo "Options: $ESBMC_PARAMS"
echo "Source: $@ \n"

for file in $SOURCES; do
   TOTAL_FILES=$((TOTAL_FILES + 1));
   FILENAME=$file

   if [ $IS_SINGLE_FILE -eq 0 ]; then
      FILENAME=$(echo $file);
      for current_source in "$@" ; do
         if [ $(echo $current_source | grep -o "\<i\>" | wc -l) -eq 1 ]; then
            continue;
         fi
         TEMP=$(echo $file | sed -e "s:$current_source::")       
  	 if [ $(expr length $TEMP) -lt $(expr length $FILENAME) ]; then 
            FILENAME=$(echo $TEMP);
         fi
      done
   fi
  
   echo "RUNNING: " $FILENAME
   
   EXPECTED_FAILED_RESULT=$(echo $file | grep 'false'| wc -l);  
   if [ $EXPECTED_FAILED_RESULT -eq 1 ]; then
      MAX_SCORE=$((MAX_SCORE + 1));
   else
      MAX_SCORE=$((MAX_SCORE + 2));
   fi   

   INITIAL_EXECUTION_TIMESTAMP=$(date +%s)
   OUT=$(sh $ESBMC_WRAPPER_SCRIPT $file;)
   FINAL_EXECUTION_TIMESTAMP=$(date +%s)

   FAILED=$(echo $OUT | grep "FALSE" | wc -l); 
   SUCCESS=$(echo $OUT | grep "TRUE" | wc -l);  
   INCORRECT_RESULT=0;  
   TIME=$((FINAL_EXECUTION_TIMESTAMP - INITIAL_EXECUTION_TIMESTAMP));
 
   CSS_CLASS="";
   RESULT_TEXT=""; 

   if [ $FAILED -eq 0 ] && [ $SUCCESS -eq 0 ]; then
      CSS_CLASS="status unknown";
      RESULT_TEXT="unknown"; 
      echo $(echo -e "\033[0;33munknown\033[0m" | cut -d " " -f2) "in $TIME""s";
      INCORRECT_RESULT=1;
   elif [ $EXPECTED_FAILED_RESULT -eq 1 ] && [ $FAILED -eq 1 ]; then
      CSS_CLASS="correctProperty";
      RESULT_TEXT="false(label)";
      CORRECT_RESULTS=$((CORRECT_RESULTS + 1));
      CORRECT_FALSES=$((CORRECT_FALSES + 1));
      echo $(echo -e "\033[0;32mfalse(label)\033[0m" | cut -d " " -f2) "in $TIME""s";
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

   HTML_ENTRY="<tr><td>$FILENAME</td><td class=\"$CSS_CLASS\">$RESULT_TEXT</td><td class=\"unknownValue\">$TIME&nbsp;</td><td style=\"display: none\">$INCORRECT_RESULT</td></tr>"
   echo $HTML_ENTRY >> $OUTPUT_REPORT_FILE
done
FINAL_TIMESTAMP=$(date +%s)

# CALCULATE POINTS
TOTAL_POINTS=$((TOTAL_POINTS + 2 * CORRECT_TRUES));
TOTAL_POINTS=$((TOTAL_POINTS + CORRECT_FALSES));
TOTAL_POINTS=$((TOTAL_POINTS - 8 * FALSE_POSITIVES));
TOTAL_POINTS=$((TOTAL_POINTS - 4 * FALSE_NEGATIVES));
TOTAL_EXECUTION_TIME=$((FINAL_TIMESTAMP - INITIAL_TIMESTAMP));

# HTML CONTENT
HTML_TABLE_FOOTER="</tbody></table><table style=\"width: 100%; margin-top: 2px\"><tfoot><tr><td style=\"width: 60%\">total files</td><td>$TOTAL_FILES</td><td class=\"unknownValue\" style=\"width: 20%\">$TOTAL_EXECUTION_TIME&nbsp;</td></tr><tr><td title=\"(no bug exists + result is SAFE) OR (bug exists + result is UNSAFE) OR (property is violated + violation is found)\">correct results</td><td>$CORRECT_RESULTS</td><td>-</td></tr><tr><td title=\"bug exists + result is SAFE\">false negatives</td><td>$FALSE_NEGATIVES</td><td>-</td></tr><tr><td title=\"no bug exists + result is UNSAFE\">false positives</td><td>$FALSE_POSITIVES</td><td>-</td></tr><tr><td title=\"17 safe files, 15 unsafe files\">score ($TOTAL_FILES files, max score: $MAX_SCORE)</td><td class=\"score\">$TOTAL_POINTS</td><td class=\"score\"></td></tr></tfoot></table></center><br><a href=\"#\" onclick=\"javascript:orderByCorrectIncorrectResults()\">order by correct / incorrect results</a></body></html>"

echo "\n*** RESULTS *** ";
echo "Total Files: $TOTAL_FILES in $TOTAL_EXECUTION_TIME""s";
echo "Correct Results: $CORRECT_RESULTS";
echo "False Negatives: $FALSE_NEGATIVES";
echo "False Positives: $FALSE_POSITIVES";
echo "Score ($TOTAL_FILES files, max score: $MAX_SCORE): $TOTAL_POINTS";

echo $HTML_TABLE_FOOTER >> $OUTPUT_REPORT_FILE
echo "\nReport file generated: $OUTPUT_REPORT_FILE";
