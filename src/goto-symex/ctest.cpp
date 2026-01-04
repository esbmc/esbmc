#include <goto-symex/ctest.h>
#include <goto-symex/slice.h>
#include <ac_config.h>
#include <util/prefix.h>
#include <util/message/format.h>
#include <irep2/irep2_expr.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <map>

std::string ctest_generator::clean_variable_name(const std::string &name) const
{
  std::string var_name = name;

  // Remove everything before the last '@' (symbol mangling)
  size_t at_pos = var_name.rfind('@');
  if (at_pos != std::string::npos)
    var_name = var_name.substr(at_pos + 1);

  // Remove everything after '!' (SSA suffix)
  size_t exclaim_pos = var_name.find('!');
  if (exclaim_pos != std::string::npos)
    var_name.resize(exclaim_pos);

  // Remove everything after '?' (other suffix)
  size_t question_pos = var_name.find('?');
  if (question_pos != std::string::npos)
    var_name.resize(question_pos);

  // Remove "c::main::" or "c::" prefix if present
  if (has_prefix(var_name, "c::main::"))
    var_name = var_name.substr(9);
  else if (has_prefix(var_name, "c::"))
    var_name = var_name.substr(3);

  // Remove remaining "::" separators (namespace/scope markers)
  size_t scope_pos;
  while ((scope_pos = var_name.find("::")) != std::string::npos)
  {
    var_name = var_name.substr(scope_pos + 2);
  }

  // Remove $tmp:: and similar internal markers
  size_t tmp_pos;
  while ((tmp_pos = var_name.find("$tmp::")) != std::string::npos)
  {
    var_name = var_name.substr(tmp_pos + 6);
  }

  // Remove $return_value$ marker
  size_t ret_pos;
  while ((ret_pos = var_name.find("return_value$")) != std::string::npos)
  {
    var_name.erase(ret_pos, 13);
  }

  // Remove any remaining $ characters (used for internal naming)
  var_name.erase(
    std::remove(var_name.begin(), var_name.end(), '$'), var_name.end());

  // If the name contains __VERIFIER_ functions, it's an internal temporary
  // Extract just a simple name or return empty to trigger fallback naming
  if (
    var_name.find("__VERIFIER_") != std::string::npos ||
    var_name.find("___VERIFIER_") != std::string::npos)
  {
    return "";
  }

  // If the name is empty or invalid, return empty for fallback naming
  if (var_name.empty() || !isalpha(var_name[0]))
    return "";

  return var_name;
}

std::string ctest_generator::extract_function_name(
  const symex_target_equationt &target,
  smt_convt &smt_conv) const
{
  // Try to extract function name from SSA steps
  for (auto const &SSA_step : target.SSA_steps)
  {
    if (!smt_conv.l_get(SSA_step.guard_ast).is_true())
      continue;

    if (SSA_step.source.pc->location.function() != "")
    {
      std::string full_func =
        SSA_step.source.pc->location.function().as_string();

      // Skip internal functions
      if (
        !has_prefix(full_func, "__ESBMC_") &&
        !has_prefix(full_func, "__VERIFIER_") && full_func != "c::__ESBMC_main")
      {
        // Clean up function name (remove "c::" prefix if present)
        if (has_prefix(full_func, "c::"))
          return full_func.substr(3);
        else
          return full_func;
      }
    }
  }

  return "main"; // Default to main
}

std::string ctest_generator::type_to_c_string(const type2tc &type) const
{
  if (is_signedbv_type(type) || is_unsignedbv_type(type))
  {
    unsigned width = type->get_width();
    if (is_signedbv_type(type))
    {
      if (width == 8)
        return "char";
      if (width == 16)
        return "short";
      if (width == 32)
        return "int";
      if (width == 64)
        return "long long";
      return "int"; // Default
    }
    else
    {
      if (width == 8)
        return "unsigned char";
      if (width == 16)
        return "unsigned short";
      if (width == 32)
        return "unsigned int";
      if (width == 64)
        return "unsigned long long";
      return "unsigned int"; // Default
    }
  }
  else if (is_floatbv_type(type))
  {
    unsigned width = type->get_width();
    if (width == 32)
      return "float";
    if (width == 64)
      return "double";
    return "double"; // Default
  }
  else if (is_bool_type(type))
  {
    return "_Bool"; // SV-COMP standard
  }
  else if (is_pointer_type(type))
  {
    return "void*";
  }

  return "int"; // Fallback
}

std::string ctest_generator::type_to_verifier_string(const type2tc &type) const
{
  if (is_signedbv_type(type) || is_unsignedbv_type(type))
  {
    unsigned width = type->get_width();
    if (is_signedbv_type(type))
    {
      if (width == 8)
        return "char";
      if (width == 16)
        return "short";
      if (width == 32)
        return "int";
      if (width == 64)
        return "long";
      return "int"; // Default
    }
    else
    {
      if (width == 8)
        return "uchar";
      if (width == 16)
        return "ushort";
      if (width == 32)
        return "uint";
      if (width == 64)
        return "ulong";
      return "uint"; // Default
    }
  }
  else if (is_floatbv_type(type))
  {
    unsigned width = type->get_width();
    if (width == 32)
      return "float";
    if (width == 64)
      return "double";
    return "float"; // Default
  }
  else if (is_bool_type(type))
  {
    return "bool";
  }
  else if (is_pointer_type(type))
  {
    return "pointer";
  }

  return "int"; // Fallback
}

std::string ctest_generator::format_c_value(const expr2tc &value) const
{
  if (is_constant_int2t(value))
  {
    return integer2string(to_constant_int2t(value).value);
  }
  else if (is_constant_floatbv2t(value))
  {
    return to_constant_floatbv2t(value).value.to_ansi_c_string();
  }
  else if (is_constant_bool2t(value))
  {
    return to_constant_bool2t(value).value ? "1" : "0";
  }

  return "0"; // Fallback
}

void ctest_generator::clear()
{
  std::lock_guard<std::mutex> lock(data_mutex);
  test_cases.clear();
  function_name.clear();
  source_file.clear();
}

void ctest_generator::collect(
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  const namespacet &ns)
{
  (void)ns;

  std::vector<test_variable> current_test;

  // Extract function name if not already set
  std::string extracted_func_name;
  if (function_name.empty())
    extracted_func_name = extract_function_name(target, smt_conv);

  // Use the SHARED collection logic from witnesses.cpp
  auto collected_values = collect_nondet_values(target, smt_conv);

  // Convert collected values to test_variable format
  for (const auto &val : collected_values)
  {
    test_variable var;
    var.verifier_type = type_to_verifier_string(val.type);
    var.c_type = type_to_c_string(val.type);
    var.value = format_c_value(val.value_expr);
    current_test.push_back(var);
  }

  // Store collected data if we found any nondet values
  if (!current_test.empty())
  {
    std::lock_guard<std::mutex> lock(data_mutex);

    // Store function name if we found one
    if (!extracted_func_name.empty() && function_name.empty())
      function_name = extracted_func_name;

    // Store source file if not set
    if (source_file.empty())
      source_file = config.options.get_option("input-file");

    test_cases.push_back(current_test);
  }
}

bool ctest_generator::has_tests() const
{
  std::lock_guard<std::mutex> lock(data_mutex);
  return !test_cases.empty();
}

void ctest_generator::generate() const
{
  std::lock_guard<std::mutex> lock(data_mutex);

  if (test_cases.empty())
  {
    log_warning("No test cases collected. No CTest files generated.");
    return;
  }

  // Generate individual test case files
  std::vector<std::string> test_file_names;

  for (size_t i = 0; i < test_cases.size(); ++i)
  {
    std::string test_file_name = "test_case_" + std::to_string(i + 1) + ".c";
    test_file_names.push_back(test_file_name);

    std::ofstream test_file(test_file_name);

    // Header
    test_file << "// Auto-generated by ESBMC " << ESBMC_VERSION << "\n";
    test_file << "// Test case " << (i + 1) << " of " << test_cases.size()
              << "\n";
    test_file << "// This file provides concrete implementations of "
                 "__VERIFIER_nondet_* functions\n\n";

    // Group variables by verifier_type
    std::map<std::string, std::vector<std::string>> type_values;
    std::map<std::string, std::string> type_c_types;

    for (const auto &var : test_cases[i])
    {
      type_values[var.verifier_type].push_back(var.value);
      type_c_types[var.verifier_type] = var.c_type;
    }

    // Generate __VERIFIER_nondet_* functions for each type
    for (const auto &[verifier_type, values] : type_values)
    {
      std::string c_type = type_c_types[verifier_type];

      test_file << c_type << " __VERIFIER_nondet_" << verifier_type
                << "(void) {\n";
      test_file << "  static int i = 0;\n";
      test_file << "  static const " << c_type << " v[] = { ";

      for (size_t j = 0; j < values.size(); ++j)
      {
        if (j > 0)
          test_file << ", ";
        test_file << values[j];
      }

      test_file << " };\n";
      test_file << "  return v[i++];\n";
      test_file << "}\n\n";
    }

    test_file.close();
  }

  // Generate CMakeLists.txt
  std::ofstream cmake_file("CMakeLists.txt");

  cmake_file << "# Auto-generated by ESBMC " << ESBMC_VERSION << "\n";
  cmake_file << "cmake_minimum_required(VERSION 3.10)\n";
  cmake_file << "project(ESBMCGeneratedTests C)\n\n";

  cmake_file << "enable_testing()\n\n";

  cmake_file << "option(ENABLE_COVERAGE \"Enable coverage reporting\" OFF)\n";
  cmake_file
    << "if(ENABLE_COVERAGE AND CMAKE_C_COMPILER_ID MATCHES \"GNU|Clang\")\n";
  cmake_file << "  add_compile_options(-O0 -g --coverage)\n";
  cmake_file << "  add_link_options(--coverage)\n";
  cmake_file << "endif()\n\n";

  // Get source file name without path
  std::string src_file_name = source_file;
  size_t last_slash = src_file_name.rfind('/');
  if (last_slash != std::string::npos)
    src_file_name = src_file_name.substr(last_slash + 1);

  cmake_file << "# Each test case is compiled with the original source + test "
                "case implementation\n";
  for (size_t i = 0; i < test_file_names.size(); ++i)
  {
    std::string test_name = "test_case_" + std::to_string(i + 1);
    cmake_file << "add_executable(" << test_name << " " << src_file_name << " "
               << test_file_names[i] << ")\n";
    cmake_file << "add_test(NAME " << test_name << " COMMAND " << test_name
               << ")\n\n";
  }

  cmake_file.close();

  log_status(
    "Generated {} CTest test case(s) with CMakeLists.txt", test_cases.size());
}

void ctest_generator::generate_single(
  const std::string &output_dir,
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  const namespacet &ns)
{
  (void)output_dir;
  (void)ns;

  // Extract source file
  std::string src_file = config.options.get_option("input-file");

  std::unordered_set<std::string> seen_nondets;
  std::vector<test_variable> test_vars;

  // Traverse SSA steps to extract nondet variables - following witnesses.cpp
  int collected_count = 0;
  int skipped_count = 0;

  for (auto const &SSA_step : target.SSA_steps)
  {
    if (!smt_conv.l_get(SSA_step.guard_ast).is_true())
      continue;

    if (SSA_step.is_assignment())
    {
      auto nondet_expr = symex_slicet::get_nondet_symbol(SSA_step.rhs);
      if (!nondet_expr || !is_symbol2t(nondet_expr))
        continue;

      const symbol2t &sym = to_symbol2t(nondet_expr);
      if (!has_prefix(sym.thename.as_string(), "nondet$"))
        continue;

      // Deduplicate by nondet symbol name (same as witnesses.cpp)
      if (seen_nondets.count(sym.thename.as_string()))
      {
        skipped_count++;
        continue;
      }

      seen_nondets.insert(sym.thename.as_string());

      // Get the concrete value from the solver
      auto concrete_value = smt_conv.get(nondet_expr);

      test_variable var;
      var.verifier_type = type_to_verifier_string(concrete_value->type);
      var.c_type = type_to_c_string(concrete_value->type);
      var.value = format_c_value(concrete_value);

      collected_count++;
      test_vars.push_back(var);
    }
  }

  if (test_vars.empty())
  {
    log_warning("No nondet variables found. No CTest test case generated.");
    return;
  }

  // Generate single test file
  std::string test_file_name = "test_case.c";
  std::ofstream test_file(test_file_name);

  test_file << "// Auto-generated by ESBMC " << ESBMC_VERSION << "\n";
  test_file << "// This file provides concrete implementations of "
               "__VERIFIER_nondet_* functions\n\n";

  // Group variables by verifier_type
  std::map<std::string, std::vector<std::string>> type_values;
  std::map<std::string, std::string> type_c_types;

  for (const auto &var : test_vars)
  {
    type_values[var.verifier_type].push_back(var.value);
    type_c_types[var.verifier_type] = var.c_type;
  }

  // Generate __VERIFIER_nondet_* functions for each type
  for (const auto &[verifier_type, values] : type_values)
  {
    std::string c_type = type_c_types[verifier_type];

    test_file << c_type << " __VERIFIER_nondet_" << verifier_type
              << "(void) {\n";
    test_file << "  static int i = 0;\n";
    test_file << "  static const " << c_type << " v[] = { ";

    for (size_t j = 0; j < values.size(); ++j)
    {
      if (j > 0)
        test_file << ", ";
      test_file << values[j];
    }

    test_file << " };\n";
    test_file << "  return v[i++];\n";
    test_file << "}\n\n";
  }

  test_file.close();

  // Generate simple CMakeLists.txt
  std::ofstream cmake_file("CMakeLists.txt");

  cmake_file << "# Auto-generated by ESBMC " << ESBMC_VERSION << "\n";
  cmake_file << "cmake_minimum_required(VERSION 3.10)\n";
  cmake_file << "project(ESBMCGeneratedTest C)\n\n";

  cmake_file << "enable_testing()\n\n";

  cmake_file << "option(ENABLE_COVERAGE \"Enable coverage reporting\" OFF)\n";
  cmake_file
    << "if(ENABLE_COVERAGE AND CMAKE_C_COMPILER_ID MATCHES \"GNU|Clang\")\n";
  cmake_file << "  add_compile_options(-O0 -g --coverage)\n";
  cmake_file << "  add_link_options(--coverage)\n";
  cmake_file << "endif()\n\n";

  // Get source file name without path
  std::string src_file_name = src_file;
  size_t last_slash = src_file_name.rfind('/');
  if (last_slash != std::string::npos)
    src_file_name = src_file_name.substr(last_slash + 1);

  cmake_file << "add_executable(test_case " << src_file_name
             << " test_case.c)\n";
  cmake_file << "add_test(NAME test_case COMMAND test_case)\n";

  cmake_file.close();

  log_status("Generated CTest test case: {}", test_file_name);
}