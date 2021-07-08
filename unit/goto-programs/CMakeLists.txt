# Apple complains about cmdline.isset and windows complains about boost
if(UNIX AND NOT APPLE)
    new_unit_test(loop-unroll-algorithms-test "loop_unroll.test.cpp" "test_goto_factory;gotoprograms;gotoalgorithms;langapi")
endif()