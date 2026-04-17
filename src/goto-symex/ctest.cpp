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

static bool is_cpp_source(const std::string &filename)
{
  size_t dot = filename.rfind('.');
  if (dot == std::string::npos)
    return false;
  std::string ext = filename.substr(dot);
  return ext == ".cpp" || ext == ".cc" || ext == ".cxx";
}

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

// File-writing: C and C++ are separated
// Maps from verifier type suffix (e.g. "int") to values / C type strings.
using type_values_t = std::map<std::string, std::vector<std::string>>;
using c_types_t = std::map<std::string, std::string>;

// Write one __VERIFIER_nondet_* implementation per entry in the maps.
// bool_keyword selects the boolean type ("_Bool" for C, "bool" for C++).
static void write_nondet_functions(
  std::ofstream &out,
  const type_values_t &type_values,
  const c_types_t &c_types,
  const std::string &bool_keyword)
{
  for (const auto &[verifier_type, values] : type_values)
  {
    std::string c_type = c_types.at(verifier_type);
    if (c_type == "_Bool")
      c_type = bool_keyword;

    out << c_type << " __VERIFIER_nondet_" << verifier_type << "(void) {\n";
    out << "  static int i = 0;\n";
    out << "  static const " << c_type << " v[] = { ";
    for (size_t j = 0; j < values.size(); ++j)
    {
      if (j > 0)
        out << ", ";
      out << values[j];
    }
    out << " };\n";
    out << "  return v[i++];\n";
    out << "}\n\n";
  }
}

// Write a C test case file to path.
static void write_c_test_file(
  const std::string &path,
  const type_values_t &type_values,
  const c_types_t &c_types,
  size_t index = 0,
  size_t total = 0)
{
  std::ofstream f(path);
  f << "// Auto-generated by ESBMC " << ESBMC_VERSION << "\n";
  if (index > 0)
    f << "// Test case " << index << " of " << total << "\n";
  f << "// This file provides concrete implementations of "
       "__VERIFIER_nondet_* functions\n\n";
  // Providing a definition here avoids linker errors.
  f << "void __VERIFIER_assume(int cond) { (void)cond; }\n\n";
  write_nondet_functions(f, type_values, c_types, "_Bool");
}

// Write a C++ test case file to path(bool is used instead of _Bool).
static void write_cpp_test_file(
  const std::string &path,
  const type_values_t &type_values,
  const c_types_t &c_types,
  size_t index = 0,
  size_t total = 0)
{
  std::ofstream f(path);
  f << "// Auto-generated by ESBMC " << ESBMC_VERSION << "\n";
  if (index > 0)
    f << "// Test case " << index << " of " << total << "\n";
  f << "// This file provides concrete implementations of "
       "__VERIFIER_nondet_* functions\n\n";
  f << "extern \"C\" {\n\n";
  f << "void __VERIFIER_assume(int cond) { (void)cond; }\n\n";
  write_nondet_functions(f, type_values, c_types, "bool");
  f << "} // extern \"C\"\n";
}

// During verification ESBMC supplies __VERIFIER_nondet_* and
// __VERIFIER_assume() internally, so source files typically contain no
// declaration for them.
// When the generated CTest cases are compiled
// thecompiler requires explicit declarations to resolve the calls in the
// original source file.
// This header is force-included by CMake via target_compile_options(-include ...)
// so no modification to the original source file is required.

// Write esbmc_verifier.h with forward declarations for all __VERIFIER_*
// functions used by the SV-COMP / ESBMC verification interface.
static void write_verifier_header()
{
  std::ofstream h("esbmc_verifier.h");
  h << "// Auto-generated by ESBMC " << ESBMC_VERSION << "\n";
  h << "//\n";
  h << "// This header declares the __VERIFIER_* functions used by the\n";
  h << "// SV-COMP / ESBMC verification interface.  It is force-included\n";
  h << "// by the generated CMakeLists.txt (via -include) so that the\n";
  h << "// original source file compiles correctly even if it does not\n";
  h << "// declare these functions itself.\n";
  h << "//\n";
  h << "// You may also add  #include \"esbmc_verifier.h\"  to your own\n";
  h << "// source file; duplicate declarations are harmless in C and C++.\n";
  h << "#pragma once\n\n";
  h << "#ifdef __cplusplus\n";
  h << "extern \"C\" {\n";
  h << "#endif\n\n";
  h << "/* Nondeterministic value functions */\n";
  h << "char __VERIFIER_nondet_char(void);\n";
  h << "short __VERIFIER_nondet_short(void);\n";
  h << "int __VERIFIER_nondet_int(void);\n";
  h << "long long __VERIFIER_nondet_long(void);\n";
  h << "unsigned char __VERIFIER_nondet_uchar(void);\n";
  h << "unsigned short __VERIFIER_nondet_ushort(void);\n";
  h << "unsigned int __VERIFIER_nondet_uint(void);\n";
  h << "unsigned long long __VERIFIER_nondet_ulong(void);\n";
  h << "float __VERIFIER_nondet_float(void);\n";
  h << "double __VERIFIER_nondet_double(void);\n";
  h << "/* In C++ 'bool' is a built-in type; in C use _Bool (native C99\n";
  h << "   type, always available without any #include). */\n";
  h << "#ifdef __cplusplus\n";
  h << "bool __VERIFIER_nondet_bool(void);\n";
  h << "#else\n";
  h << "_Bool __VERIFIER_nondet_bool(void);\n";
  h << "#endif\n";
  h << "void *__VERIFIER_nondet_pointer(void);\n\n";
  h << "/* Assumption function */\n";
  h << "void __VERIFIER_assume(int cond);\n\n";
  h << "#ifdef __cplusplus\n";
  h << "} /* extern \"C\" */\n";
  h << "#endif\n";
}

static std::string file_basename(const std::string &path)
{
  size_t slash = path.rfind('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

// Write CMakeLists.txt for a C project.
// Generates esbmc_verifier.h and force-includes it via target_compile_options.
// Ensure the original source file compiles correctly.
static void write_cmake(
  bool cpp_mode,
  const std::string &project_name,
  const std::string &src_file,
  const std::vector<std::string> &target_names,
  const std::vector<std::string> &test_files)
{
  write_verifier_header();

  const std::string lang = cpp_mode ? "CXX" : "C";
  const std::string compiler_id =
    cpp_mode ? "CMAKE_CXX_COMPILER_ID" : "CMAKE_C_COMPILER_ID";

  std::string src = file_basename(src_file);
  std::ofstream cmake("CMakeLists.txt");
  cmake << "# Auto-generated by ESBMC " << ESBMC_VERSION << "\n";
  cmake << "cmake_minimum_required(VERSION 3.10)\n";
  cmake << "project(" << project_name << " " << lang << ")\n\n";
  cmake << "enable_testing()\n\n";
  cmake << "option(ENABLE_COVERAGE \"Enable coverage reporting\" OFF)\n";
  cmake << "if(ENABLE_COVERAGE AND " << compiler_id
        << " MATCHES \"GNU|Clang\")\n";
  cmake << "  add_compile_options(-O0 -g --coverage)\n";
  cmake << "  add_link_options(--coverage)\n";
  cmake << "endif()\n\n";
  cmake << "# Each test case is compiled with the original source + test case "
           "implementation\n";
  for (size_t i = 0; i < target_names.size(); ++i)
  {
    cmake << "add_executable(" << target_names[i] << " " << src << " "
          << test_files[i] << ")\n";
    cmake << "add_test(NAME " << target_names[i] << " COMMAND "
          << target_names[i] << ")\n";
    // Force-include esbmc_verifier.h so the source file gets __VERIFIER_*
    // declarations even if it does not declare them itself.
    cmake << "target_compile_options(" << target_names[i]
          << " PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/esbmc_verifier.h)"
             "\n\n";
  }
}

void ctest_generator::generate() const
{
  std::lock_guard<std::mutex> lock(data_mutex);

  if (test_cases.empty())
  {
    log_warning("No test cases collected. No CTest files generated.");
    return;
  }

  bool cpp_mode = is_cpp_source(source_file);
  std::string test_ext = cpp_mode ? ".cpp" : ".c";

  // Deduplicate detect
  // two test cases are identical when their (type, value)
  // sequences are equal.
  // Build a fingerprint string and skip duplicates.
  auto fingerprint = [](const std::vector<test_variable> &tc) {
    std::string fp;
    for (const auto &v : tc)
    {
      fp += v.verifier_type;
      fp += '=';
      fp += v.value;
      fp += ';';
    }
    return fp;
  };

  std::unordered_set<std::string> seen;
  std::vector<size_t> unique_indices;
  for (size_t j = 0; j < test_cases.size(); ++j)
  {
    if (seen.insert(fingerprint(test_cases[j])).second)
      unique_indices.push_back(j);
  }

  size_t skipped = test_cases.size() - unique_indices.size();
  if (skipped > 0)
    log_status("Skipped {} duplicate test case(s).", skipped);

  std::vector<std::string> target_names;
  std::vector<std::string> test_file_names;

  for (size_t i = 0; i < unique_indices.size(); ++i)
  {
    const auto &tc = test_cases[unique_indices[i]];
    std::string target = "test_case_" + std::to_string(i + 1);
    std::string test_file = target + test_ext;
    target_names.push_back(target);
    test_file_names.push_back(test_file);

    // Build type maps from the collected test variables.
    type_values_t type_values;
    c_types_t c_types;
    for (const auto &var : tc)
    {
      type_values[var.verifier_type].push_back(var.value);
      c_types[var.verifier_type] = var.c_type;
    }

    if (cpp_mode)
      write_cpp_test_file(
        test_file, type_values, c_types, i + 1, unique_indices.size());
    else
      write_c_test_file(
        test_file, type_values, c_types, i + 1, unique_indices.size());
  }

  write_cmake(
    cpp_mode,
    "ESBMCGeneratedTests",
    source_file,
    target_names,
    test_file_names);

  log_status(
    "Generated {} CTest test case(s) with CMakeLists.txt",
    unique_indices.size());
}

void ctest_generator::generate_single(
  const std::string &output_dir,
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  const namespacet &ns)
{
  (void)output_dir;
  (void)ns;

  std::string src_file = config.options.get_option("input-file");

  std::unordered_set<std::string> seen_nondets;
  std::vector<test_variable> test_vars;

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

      if (seen_nondets.count(sym.thename.as_string()))
        continue;
      seen_nondets.insert(sym.thename.as_string());

      auto concrete_value = smt_conv.get(nondet_expr);

      test_variable var;
      var.verifier_type = type_to_verifier_string(concrete_value->type);
      var.c_type = type_to_c_string(concrete_value->type);
      var.value = format_c_value(concrete_value);
      test_vars.push_back(var);
    }
  }

  if (test_vars.empty())
  {
    log_warning("No nondet variables found. No CTest test case generated.");
    return;
  }

  bool cpp_mode = is_cpp_source(src_file);
  std::string test_ext = cpp_mode ? ".cpp" : ".c";

  // Build type maps from the extracted test variables.
  type_values_t type_values;
  c_types_t c_types;
  for (const auto &var : test_vars)
  {
    type_values[var.verifier_type].push_back(var.value);
    c_types[var.verifier_type] = var.c_type;
  }

  std::string test_file_name = "test_case" + test_ext;

  if (cpp_mode)
    write_cpp_test_file(test_file_name, type_values, c_types);
  else
    write_c_test_file(test_file_name, type_values, c_types);
  write_cmake(
    cpp_mode, "ESBMCGeneratedTest", src_file, {"test_case"}, {test_file_name});

  log_status("Generated CTest test case: {}", test_file_name);
}