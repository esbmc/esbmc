if(NOT WIN32)
new_unit_test(python_annotation_test "python_annotation_test.cpp" "pythonfrontend;util_esbmc;bigint;nlohmann_json::nlohmann_json")
endif()
new_unit_test(symbol_id_test "symbol_id_test.cpp" "pythonfrontend")
new_unit_test(python_class_test "python_class_test.cpp" "pythonfrontend")
new_unit_test(complex_type_test "complex_type_test.cpp" "pythonfrontend;util_esbmc;bigint")
new_unit_test(irep2_type_roundtrip_test "irep2_type_roundtrip_test.cpp" "pythonfrontend;util_esbmc;bigint")
new_unit_test(python_adjust_test "python_adjust_test.cpp" "pythonfrontend;util_esbmc;bigint")
new_unit_test(function_call_cache_test "function_call_cache_test.cpp" "pythonfrontend")
new_unit_test(json_utils_alias_test "json_utils_alias_test.cpp" "pythonfrontend")
if(NOT WIN32)
new_unit_test(function_call_expr_error_test "function_call_expr_error_test.cpp" "pythonfrontend;util_esbmc;bigint")
endif()

# Differential model tests: a pytest/hypothesis suite (test_*.py in this directory)
# that loads the Python operational models in src/python-frontend/models and the
# preprocessor/parser packages, comparing their behaviour against CPython. It is pure
# Python — no ESBMC binary required — so it runs as a single standalone ctest.
# Gated on pytest + hypothesis being importable (CI installs them via scripts/build.sh);
# skipped with a STATUS message otherwise so a minimal build still configures.
if(DEFINED Python3_EXECUTABLE)
  set(_python_model_diff_interp ${Python3_EXECUTABLE})
else()
  find_package(Python COMPONENTS Interpreter)
  set(_python_model_diff_interp ${Python_EXECUTABLE})
endif()

if(_python_model_diff_interp)
  execute_process(
    COMMAND ${_python_model_diff_interp} -c "import pytest, hypothesis"
    RESULT_VARIABLE _python_model_diff_missing_deps
    OUTPUT_QUIET ERROR_QUIET)
  if(_python_model_diff_missing_deps EQUAL 0)
    add_test(NAME python_model_diff
             COMMAND ${_python_model_diff_interp} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(python_model_diff PROPERTIES LABELS "python")
  else()
    message(STATUS "python_model_diff: pytest/hypothesis not found, skipping")
  endif()
else()
  message(STATUS "python_model_diff: no Python interpreter found, skipping")
endif()
