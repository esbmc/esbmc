new_unit_test(property-report-test "property_report.test.cpp" "esbmc-driver;mode_table_obj")
new_unit_test(bmc-run-test "bmc_run.test.cpp" "esbmc-driver;test_goto_factory;clangcppfrontend;clibs;filesystem;langapi;build_id_obj;gotoprograms;$<LINK_LIBRARY:WHOLE_ARCHIVE,gotoalgorithms>")
