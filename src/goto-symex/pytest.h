#ifndef CPROVER_GOTO_SYMEX_PYTEST_H
#define CPROVER_GOTO_SYMEX_PYTEST_H

#include <goto-symex/symex_target_equation.h>
#include <solvers/smt/smt_conv.h>
#include <util/namespace.h>
#include <mutex>
#include <string>
#include <vector>
#include <fstream>

/// Generates pytest test-cases for Python programs
class pytest_generator
{
private:
  std::vector<std::vector<std::string>> test_cases;
  std::vector<std::string> param_names;
  std::string function_name;
  mutable std::mutex data_mutex;

  /// Clean up ESBMC internal variable names
  std::string clean_variable_name(const std::string &name) const;

  /// Extract function name from SSA steps
  std::string extract_function_name(
    const symex_target_equationt &target,
    smt_convt &smt_conv) const;

  /// Convert C-style float string to Python format
  std::string convert_float_to_python(const std::string &c_float) const;

  /// Escape string for Python representation
  std::string escape_python_string(const std::string &str) const;

  /// TODO: Convert array to Python list representation
  std::string convert_array_to_python_list(const expr2tc &array_expr) const;

  /// TODO: Convert struct to Python dict representation
  std::string convert_struct_to_python_dict(const expr2tc &struct_expr) const;

  /// Check if array is a character array (C string)
  bool is_char_array(const expr2tc &array_expr) const;

  /// Convert character array to Python string
  std::string convert_char_array_to_string(const expr2tc &array_expr) const;

  /// Write pytest file header
  void write_file_header(std::ofstream &file, const std::string &original_file)
    const;

  /// Write import statements
  void write_imports(std::ofstream &file, const std::string &module_name) const;

  /// Build comma-separated parameter list string
  static std::string build_param_list(const std::vector<std::string> &params);

  /// Write test data in parametrize format
  void write_test_data(
    std::ofstream &file,
    const std::vector<std::string> &param_names,
    const std::vector<std::vector<std::string>> &test_data) const;

  /// Write test function definition
  void write_test_function(
    std::ofstream &file,
    const std::string &func_name,
    const std::vector<std::string> &param_names) const;

public:
  pytest_generator() = default;

  /// Extract module name from input file path and removes .py extension and directory
  static std::string extract_module_name(const std::string &input_file);

  /// Generate pytest filename from module name
  static std::string generate_pytest_filename(const std::string &module_name);

  /// Clear collected data (called at start of coverage run)
  void clear();

  /// Collect test data from a counterexample (called for each CEX in coverage mode)
  void collect(const symex_target_equationt &target, smt_convt &smt_conv);

  /// Generate pytest file from collected data (called at end of coverage mode)
  void generate(const std::string &file_name) const;

  /// Single-shot generation for non-coverage mode
  void generate_single(
    const std::string &file_name,
    const symex_target_equationt &target,
    smt_convt &smt_conv,
    const namespacet &ns);

  /// Check if any test cases have been collected
  bool has_tests() const;
};

#endif