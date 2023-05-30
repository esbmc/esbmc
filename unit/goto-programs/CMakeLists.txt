new_unit_test(loop-unroll-algorithms-test "loop_unroll.test.cpp" "test_goto_factory;gotoprograms;gotoalgorithms;filesystem;langapi")
new_unit_test(interval-template-test "interval_template.test.cpp" "gotoprograms")
if(NOT WIN32)    
   # new_unit_test(interval-analysis-test "interval_analysis.test.cpp" "test_goto_factory;gotoprograms;gotoalgorithms;filesystem;langapi")
else()
    message(WARNING "For some reason Interval Analysis tests are failing on windows")
endif()

