# Apple complains about cmdline.isset and windows complains about boost
if(UNIX AND NOT APPLE)
    new_unit_test(goto-algorithms-test "goto-algorithms.test.cpp" "test_goto_factory;goto_algorithms;gotoprograms;langapi")
endif()