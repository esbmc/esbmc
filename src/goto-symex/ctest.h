#ifndef CPROVER_GOTO_SYMEX_CTEST_H
#define CPROVER_GOTO_SYMEX_CTEST_H

#include <goto-symex/symex_target_equation.h>
#include <solvers/smt/smt_conv.h>
#include <util/namespace.h>
#include <string>
#include <vector>
#include <mutex>
#include "witnesses.h"

/// This generates CTest test-cases for C programs
class ctest_generator
{
private:
  struct test_variable
  {
    std::string
      verifier_type; // "int", "uint", "char", "uchar", "float", etc. for __VERIFIER_nondet_TYPE
    std::string c_type; // C type: "int", "unsigned int", "float", etc.
    std::string value;  // The concrete value
  };

  std::vector<std::vector<test_variable>> test_cases;
  std::string function_name;
  std::string source_file;
  mutable std::mutex data_mutex;

  /// Clean up ESBMC internal variable names
  std::string clean_variable_name(const std::string &name) const;

  /// Extract function name from SSA steps
  std::string extract_function_name(
    const symex_target_equationt &target,
    smt_convt &smt_conv) const;

  /// Convert type2t to C type string and VERIFIER type string
  std::string type_to_c_string(const type2tc &type) const;
  std::string type_to_verifier_string(const type2tc &type) const;

  /// Format value for C code
  std::string format_c_value(const expr2tc &value) const;

public:
  ctest_generator() = default;

  /// Clear collected data (called at start of coverage run)
  void clear();

  /// Collect test data from a counterexample (called for each CEX in coverage mode)
  void collect(
    const symex_target_equationt &target,
    smt_convt &smt_conv,
    const namespacet &ns);

  /// Generate C test files and CMakeLists.txt from collected data
  void generate() const;

  /// Single-shot generation for non-coverage mode
  void generate_single(
    const std::string &output_dir,
    const symex_target_equationt &target,
    smt_convt &smt_conv,
    const namespacet &ns);

  /// Check if any test cases have been collected
  bool has_tests() const;
};

#endif