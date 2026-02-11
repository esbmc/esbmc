#include <goto-symex/pytest.h>
#include <goto-symex/slice.h>
#include <ac_config.h>
#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <util/c_types.h>
#include <util/message.h>
#include <util/config.h>
#include <util/string_constant.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>

// pytest_generator class implementation
std::string pytest_generator::extract_module_name(const std::string &input_file)
{
  std::string module_name = input_file;

  // Remove .py extension
  size_t dot_pos = module_name.rfind(".py");
  if (dot_pos != std::string::npos && dot_pos < module_name.size())
    module_name.resize(dot_pos);

  // Remove directory path
  size_t slash_pos = module_name.rfind("/");
  if (slash_pos != std::string::npos)
    module_name = module_name.substr(slash_pos + 1);

  return module_name;
}

std::string
pytest_generator::generate_pytest_filename(const std::string &module_name)
{
  return "test_" + module_name + ".py";
}

std::string pytest_generator::clean_variable_name(const std::string &name) const
{
  std::string var_name = name;

  // Remove everything before the last '$' (if present) - for internal symbols like "$nondet_str$15"
  size_t dollar_pos = var_name.rfind('$');
  if (dollar_pos != std::string::npos && dollar_pos > 0)
  {
    // Check if this looks like an internal symbol (contains multiple $)
    size_t first_dollar = var_name.find('$');
    if (first_dollar != dollar_pos)
    {
      // Multiple $ signs - this is likely an internal symbol, extract the meaningful part
      // For "python_converter::test.py:9$nondet_str$15", extract "nondet_str"
      size_t start = first_dollar + 1;
      size_t end = var_name.rfind('$');
      if (end > start)
        var_name = var_name.substr(start, end - start);
    }
  }

  // Remove everything before the last '@' (Python mangling)
  size_t at_pos = var_name.rfind('@');
  if (at_pos != std::string::npos)
    var_name = var_name.substr(at_pos + 1);

  // Remove everything after '!' (SSA suffix)
  size_t exclaim_pos = var_name.find('!');
  if (exclaim_pos != std::string::npos)
    if (exclaim_pos < var_name.size())
      var_name.resize(exclaim_pos);

  // Remove everything after '?' (other suffix)
  size_t question_pos = var_name.find('?');
  if (question_pos != std::string::npos)
    if (question_pos < var_name.size())
      var_name.resize(question_pos);

  // Remove "c::main::" prefix if present
  if (has_prefix(var_name, "c::main::"))
    var_name = var_name.substr(9);

  // Remove "python_converter::" prefix if present
  size_t converter_pos = var_name.find("python_converter::");
  if (converter_pos != std::string::npos)
    var_name = var_name.substr(converter_pos + 18);

  // Remove file paths and line numbers (e.g., "test.py:9:")
  size_t colon_pos = var_name.find(".py:");
  if (colon_pos != std::string::npos)
  {
    // Find the next part after the line number
    size_t next_part = var_name.find(':', colon_pos + 4);
    if (next_part != std::string::npos)
      var_name = var_name.substr(next_part + 1);
  }

  return var_name;
}

std::string pytest_generator::extract_function_name(
  const symex_target_equationt &target,
  smt_convt &smt_conv) const
{
  // List of functions to skip (C library and ESBMC internal functions)
  static const std::vector<std::string> skip_functions = {
    "strcmp",
    "strlen",
    "strcpy",
    "strcat",
    "memcpy",
    "memset",
    "memmove",
    "malloc",
    "calloc",
    "realloc",
    "free",
    "printf",
    "scanf",
    "sprintf",
    "snprintf",
    "__ESBMC_main",
    "python_user_main",
    "python_init",
    // nondet helper functions
    "nondet_list",
    "nondet_dict",
    "_nondet_size",
  };

  std::string best_candidate;

  // extract function name from SSA steps
  for (auto const &SSA_step : target.SSA_steps)
  {
    if (!smt_conv.l_get(SSA_step.guard_ast).is_true())
      continue;

    if (SSA_step.source.pc->location.function() != "")
    {
      std::string full_func =
        SSA_step.source.pc->location.function().as_string();

      // Remove "c::" prefix for comparison
      std::string func_to_check = full_func;
      if (has_prefix(func_to_check, "c::"))
        func_to_check = func_to_check.substr(3);

      // Skip internal functions
      if (
        has_prefix(full_func, "python_") || has_prefix(full_func, "__ESBMC_") ||
        has_prefix(full_func, "__VERIFIER_") ||
        has_prefix(func_to_check, "nondet_") ||
        has_prefix(func_to_check, "__") ||  // Skip all double-underscore functions
        func_to_check == "_nondet_size")
        continue;

      // Skip C library functions
      bool should_skip = false;
      for (const auto &skip_func : skip_functions)
      {
        if (func_to_check == skip_func || full_func == "c::" + skip_func)
        {
          should_skip = true;
          break;
        }
      }

      if (should_skip)
        continue;

      // This looks like a user function - save it
      // Prefer the first non-internal function we find
      if (best_candidate.empty())
      {
        // Clean up function name (remove "c::" prefix if present)
        if (has_prefix(full_func, "c::"))
          best_candidate = full_func.substr(3);
        else
          best_candidate = full_func;
      }
    }
  }

  return best_candidate;
}

std::string
pytest_generator::convert_float_to_python(const std::string &c_float) const
{
  // Convert C-style float representations to Python format
  // C style: +NAN, -NAN, +INF, -INF, +INFINITY, -INFINITY
  // Python: float('nan'), float('inf'), float('-inf')

  std::string upper = c_float;
  // Convert to uppercase for case-insensitive comparison
  for (char &c : upper)
    c = std::toupper(c);

  // Check for NaN
  if (upper.find("NAN") != std::string::npos)
    return "float('nan')";

  // Check for positive infinity
  if (
    upper.find("+INF") != std::string::npos ||
    (upper.find("INF") != std::string::npos && upper[0] != '-'))
    return "float('inf')";

  // Check for negative infinity
  if (upper.find("-INF") != std::string::npos)
    return "float('-inf')";

  // Regular float - return as-is
  return c_float;
}

std::string pytest_generator::escape_python_string(const std::string &str) const
{
  std::string result;
  result.reserve(str.size() + 10); // Reserve extra space for escapes

  for (char c : str)
  {
    switch (c)
    {
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '\"':
      result += "\\\"";
      break;
    case '\'':
      result += "\\'";
      break;
    case '\0':
      result += "\\0";
      break;
    default:
      // For printable ASCII characters, add as-is
      if (c >= 32 && c <= 126)
      {
        result += c;
      }
      else
      {
        // For non-printable characters, use hex escape
        char buf[5];
        snprintf(buf, sizeof(buf), "\\x%02x", static_cast<unsigned char>(c));
        result += buf;
      }
      break;
    }
  }

  return result;
}

bool pytest_generator::is_char_array(const expr2tc &array_expr) const
{
  if (!is_constant_array2t(array_expr))
    return false;

  const constant_array2t &arr = to_constant_array2t(array_expr);

  // Check if all elements are integers (char values) in the valid range
  // and the array ends with a null terminator (0)
  if (arr.datatype_members.empty())
    return false;

  // Check if the last element is 0 (null terminator)
  const expr2tc &last_elem = arr.datatype_members.back();
  if (is_constant_int2t(last_elem))
  {
    BigInt value = to_constant_int2t(last_elem).value;
    if (value == 0)
    {
      // All elements should be int (char) type
      for (const auto &elem : arr.datatype_members)
      {
        if (!is_constant_int2t(elem))
          return false;
      }
      return true;
    }
  }

  return false;
}

std::string
pytest_generator::convert_char_array_to_string(const expr2tc &array_expr) const
{
  if (!is_constant_array2t(array_expr))
    return "\"\"";

  const constant_array2t &arr = to_constant_array2t(array_expr);
  std::string result;

  // Convert character array to string (stop at null terminator or -1)
  for (const auto &elem : arr.datatype_members)
  {
    if (!is_constant_int2t(elem))
      continue;

    BigInt value = to_constant_int2t(elem).value;

    // Stop at null terminator or invalid chars
    if (value == 0 || value < 0)
      break;

    // Only include valid ASCII/UTF-8 characters
    if (value > 0 && value <= 127)
    {
      result += static_cast<char>(value.to_int64());
    }
  }

  // Escape the string and wrap in quotes
  return "\"" + escape_python_string(result) + "\"";
}

std::string
pytest_generator::convert_array_to_python_list(const expr2tc &array_expr) const
{
  if (!is_constant_array2t(array_expr))
    return "[]"; // Return empty list for non-array types

  const constant_array2t &arr = to_constant_array2t(array_expr);
  std::string result = "[";

  bool first = true;
  for (const auto &elem : arr.datatype_members)
  {
    if (!first)
      result += ", ";
    first = false;

    // Recursively convert each element based on its type
    if (is_constant_int2t(elem))
    {
      result += integer2string(to_constant_int2t(elem).value);
    }
    else if (is_constant_floatbv2t(elem))
    {
      std::string c_float =
        to_constant_floatbv2t(elem).value.to_ansi_c_string();
      result += convert_float_to_python(c_float);
    }
    else if (is_constant_bool2t(elem))
    {
      result += to_constant_bool2t(elem).value ? "True" : "False";
    }
    else if (is_constant_string2t(elem))
    {
      std::string raw_str = to_constant_string2t(elem).value.as_string();
      result += "\"" + escape_python_string(raw_str) + "\"";
    }
    else if (is_constant_array2t(elem))
    {
      // Nested list
      result += convert_array_to_python_list(elem);
    }
    else if (is_constant_struct2t(elem))
    {
      // Nested dict
      result += convert_struct_to_python_dict(elem);
    }
    else
    {
      // Unsupported type - use None
      result += "None";
    }
  }

  result += "]";
  return result;
}

std::string pytest_generator::convert_struct_to_python_dict(
  const expr2tc &struct_expr) const
{
  if (!is_constant_struct2t(struct_expr))
    return "{}"; // Return empty dict for non-struct types

  const constant_struct2t &strct = to_constant_struct2t(struct_expr);

  // Python dicts in ESBMC are represented as structs with two members:
  // - keys (list)
  // - values (list)
  if (strct.datatype_members.size() != 2)
  {
    // Not a Python dict structure, return empty dict
    return "{}";
  }

  const expr2tc &keys_expr = strct.datatype_members[0];
  const expr2tc &values_expr = strct.datatype_members[1];

  // Both keys and values should be arrays
  if (!is_constant_array2t(keys_expr) || !is_constant_array2t(values_expr))
    return "{}";

  const constant_array2t &keys = to_constant_array2t(keys_expr);
  const constant_array2t &values = to_constant_array2t(values_expr);

  // Keys and values should have the same length
  if (keys.datatype_members.size() != values.datatype_members.size())
    return "{}";

  std::string result = "{";
  bool first = true;

  for (size_t i = 0; i < keys.datatype_members.size(); ++i)
  {
    if (!first)
      result += ", ";
    first = false;

    const expr2tc &key = keys.datatype_members[i];
    const expr2tc &value = values.datatype_members[i];

    // Convert key
    std::string key_str;
    if (is_constant_int2t(key))
    {
      key_str = integer2string(to_constant_int2t(key).value);
    }
    else if (is_constant_string2t(key))
    {
      std::string raw_str = to_constant_string2t(key).value.as_string();
      key_str = "\"" + escape_python_string(raw_str) + "\"";
    }
    else if (is_constant_bool2t(key))
    {
      key_str = to_constant_bool2t(key).value ? "True" : "False";
    }
    else
    {
      key_str = "None";
    }

    // Convert value
    std::string value_str;
    if (is_constant_int2t(value))
    {
      value_str = integer2string(to_constant_int2t(value).value);
    }
    else if (is_constant_floatbv2t(value))
    {
      std::string c_float =
        to_constant_floatbv2t(value).value.to_ansi_c_string();
      value_str = convert_float_to_python(c_float);
    }
    else if (is_constant_bool2t(value))
    {
      value_str = to_constant_bool2t(value).value ? "True" : "False";
    }
    else if (is_constant_string2t(value))
    {
      std::string raw_str = to_constant_string2t(value).value.as_string();
      value_str = "\"" + escape_python_string(raw_str) + "\"";
    }
    else if (is_constant_array2t(value))
    {
      // Nested list
      value_str = convert_array_to_python_list(value);
    }
    else if (is_constant_struct2t(value))
    {
      // Nested dict
      value_str = convert_struct_to_python_dict(value);
    }
    else
    {
      value_str = "None";
    }

    result += key_str + ": " + value_str;
  }

  result += "}";
  return result;
}

void pytest_generator::write_file_header(
  std::ofstream &file,
  const std::string &original_file) const
{
  file << "# Auto-generated by ESBMC " << ESBMC_VERSION << "\n";
  file << "# Original file: " << original_file << "\n";
  file << "# Generated: "
       << boost::posix_time::to_simple_string(
            boost::posix_time::microsec_clock::universal_time())
       << "\n\n";
}

void pytest_generator::write_imports(
  std::ofstream &file,
  const std::string &module_name) const
{
  file << "from " << module_name << " import *\n";
  file << "import pytest\n\n";
}

std::string
pytest_generator::build_param_list(const std::vector<std::string> &params)
{
  std::string param_list;
  for (size_t i = 0; i < params.size(); ++i)
  {
    if (i > 0)
      param_list += ",";
    param_list += params[i];
  }
  return param_list;
}

void pytest_generator::write_test_data(
  std::ofstream &file,
  const std::vector<std::string> &param_names,
  const std::vector<std::vector<std::string>> &test_data) const
{
  std::string param_list = build_param_list(param_names);
  file << "@pytest.mark.parametrize(\"" << param_list << "\", [\n";

  bool single_param = (param_names.size() == 1);

  for (const auto &test_case : test_data)
  {
    if (single_param)
    {
      file << "    " << test_case[0] << ",\n";
    }
    else
    {
      file << "    (";
      for (size_t i = 0; i < test_case.size(); ++i)
      {
        if (i > 0)
          file << ", ";
        file << test_case[i];
      }
      file << "),\n";
    }
  }

  file << "])\n";
}

void pytest_generator::write_test_function(
  std::ofstream &file,
  const std::string &func_name,
  const std::vector<std::string> &param_names) const
{
  std::string param_list = build_param_list(param_names);

  file << "def test_" << func_name << "(" << param_list << "):\n";
  file << "    \"\"\"Auto-generated test cases for " << func_name << "\"\"\"\n";

  if (!func_name.empty() && func_name != "coverage")
  {
    // Generate actual function call
    file << "    " << func_name << "(" << param_list << ")\n\n";
  }
  else
  {
    // Fallback if we couldn't determine the function name
    file << "    # Could not determine function name\n";
    file << "    pass\n\n";
  }
}

void pytest_generator::clear()
{
  std::lock_guard<std::mutex> lock(data_mutex);
  test_cases.clear();
  param_names.clear();
  function_name.clear();
}

void pytest_generator::collect(
  const symex_target_equationt &target,
  smt_convt &smt_conv)
{
  std::vector<std::string> current_params;
  std::vector<std::string> current_param_names;
  std::unordered_set<std::string> seen_nondets;

  // Track nondet_list/nondet_dict components for building composite values
  // We collect size values and element values separately, then combine them
  std::vector<std::pair<std::string, BigInt>> list_sizes;   // (nondet_symbol, size_value)
  std::vector<std::pair<std::string, std::string>> list_elems; // (nondet_symbol, elem_value_str)
  std::vector<std::pair<std::string, std::string>> dict_keys;  // (nondet_symbol, key_value_str)
  std::vector<std::pair<std::string, std::string>> dict_values; // (nondet_symbol, value_value_str)

  // Extract function name if not already set
  std::string extracted_func_name;
  if (function_name.empty())
    extracted_func_name = extract_function_name(target, smt_conv);

  // Extract nondet values from counterexample
  int checked_assignments = 0;
  int found_nondets = 0;

  for (auto const &SSA_step : target.SSA_steps)
  {
    if (!smt_conv.l_get(SSA_step.guard_ast).is_true())
      continue;

    if (SSA_step.is_assignment())
    {
      checked_assignments++;

      // Extract variable name first
      std::string var_name;
      if (is_symbol2t(SSA_step.lhs))
      {
        const symbol2t &lhs_sym = to_symbol2t(SSA_step.lhs);
        var_name = clean_variable_name(lhs_sym.get_symbol_name());
      }

      // Track nondet_list/nondet_dict internal variables to build composite values
      // We need to collect size, elem_type, key_type, value_type to reconstruct lists/dicts
      bool is_list_size = false;
      bool is_list_elem = false;
      bool is_dict_key = false;
      bool is_dict_value = false;

      if (SSA_step.source.pc->location.function() != "")
      {
        std::string func_name =
          SSA_step.source.pc->location.function().as_string();

        // Check if we're inside an internal function
        bool in_nondet_size = false;
        bool in_nondet_list = false;
        bool in_nondet_dict = false;

        if (func_name == "_nondet_size")
          in_nondet_size = true;
        else if (func_name == "nondet_list")
          in_nondet_list = true;
        else if (func_name == "nondet_dict")
          in_nondet_dict = true;
        else
        {
          // Check for mangled names
          size_t sep_pos = func_name.rfind("::");
          if (sep_pos != std::string::npos)
          {
            std::string base_name = func_name.substr(sep_pos + 2);
            if (base_name == "_nondet_size")
              in_nondet_size = true;
            else if (base_name == "nondet_list")
              in_nondet_list = true;
            else if (base_name == "nondet_dict")
              in_nondet_dict = true;
          }
        }

        // Track the role of internal variables
        // Check based on function context first
        if (in_nondet_size && var_name == "size")
        {
          is_list_size = true;
        }
        else if (in_nondet_list)
        {
          if (var_name == "elem_type" || var_name == "elem")
            is_list_elem = true;
        }
        else if (in_nondet_dict)
        {
          if (var_name == "key_type" || var_name == "k")
            is_dict_key = true;
          else if (var_name == "value_type" || var_name == "v")
            is_dict_value = true;
        }

        // Skip only loop counters and result variables (not nondet values)
        if (
          (in_nondet_list || in_nondet_dict || in_nondet_size) &&
          (var_name == "i" || var_name == "result"))
        {
          continue;
        }
      }

      // Also check variable names outside internal functions
      // This handles cases where user passes nondet_int() as argument:
      // e.g., nondet_dict(2, key_type=nondet_int(), value_type=nondet_int())
      // In this case, the nondet assignment happens in user code, not in nondet_dict
      if (!is_list_size && !is_list_elem && !is_dict_key && !is_dict_value)
      {
        if (var_name == "key_type")
          is_dict_key = true;
        else if (var_name == "value_type")
          is_dict_value = true;
        else if (var_name == "elem_type")
          is_list_elem = true;
      }

      // Check if this is a nondet assignment
      auto nondet_expr = symex_slicet::get_nondet_symbol(SSA_step.rhs);
      if (!nondet_expr || !is_symbol2t(nondet_expr))
        continue;

      const symbol2t &sym = to_symbol2t(nondet_expr);
      if (!has_prefix(sym.thename.as_string(), "nondet$"))
        continue;

      // For dict/list components, allow duplicate nondet symbols
      // (key_type and value_type may share the same nondet symbol due to solver optimization)
      bool is_component = is_list_size || is_list_elem || is_dict_key || is_dict_value;

      if (seen_nondets.count(sym.thename.as_string()) && !is_component)
        continue;

      if (!seen_nondets.count(sym.thename.as_string()))
      {
        seen_nondets.insert(sym.thename.as_string());
        found_nondets++;
      }

      // Get concrete value
      auto concrete_value = smt_conv.get(nondet_expr);
      std::string value_str;

      if (is_constant_int2t(concrete_value))
      {
        BigInt int_value = to_constant_int2t(concrete_value).value;
        value_str = integer2string(int_value);

        // If this is a list/dict size, store it for later use
        if (is_list_size)
        {
          list_sizes.push_back({sym.thename.as_string(), int_value});
          continue; // Don't add as regular param - will be combined later
        }
      }
      else if (is_constant_floatbv2t(concrete_value))
      {
        std::string c_float =
          to_constant_floatbv2t(concrete_value).value.to_ansi_c_string();
        value_str = convert_float_to_python(c_float);
      }
      else if (is_constant_bool2t(concrete_value))
        value_str = to_constant_bool2t(concrete_value).value ? "True" : "False";
      else if (is_constant_string2t(concrete_value))
      {
        std::string raw_str =
          to_constant_string2t(concrete_value).value.as_string();
        value_str = "\"" + escape_python_string(raw_str) + "\"";
      }
      else if (is_constant_array2t(concrete_value))
      {
        // Check if this is a character array (string) first
        if (is_char_array(concrete_value))
        {
          // Python string from char array (nondet_str)
          value_str = convert_char_array_to_string(concrete_value);
        }
        else
        {
          // Python list (nondet_list)
          value_str = convert_array_to_python_list(concrete_value);
        }
      }
      else if (is_constant_struct2t(concrete_value))
      {
        // Python dict (nondet_dict)
        value_str = convert_struct_to_python_dict(concrete_value);
      }
      else
        continue; // Skip unsupported types

      // Store list/dict element values for later combination
      if (is_list_elem)
      {
        list_elems.push_back({sym.thename.as_string(), value_str});
        continue; // Don't add as regular param - will be combined later
      }
      if (is_dict_key)
      {
        dict_keys.push_back({sym.thename.as_string(), value_str});
        continue;
      }
      if (is_dict_value)
      {
        dict_values.push_back({sym.thename.as_string(), value_str});
        continue;
      }

      // Handle duplicate parameter names by adding numeric suffix
      std::string final_var_name = var_name;
      if (
        var_name.empty() || var_name == "nondet_str" ||
        var_name == "nondet_list" || var_name == "nondet_dict" ||
        var_name == "nondet_int" || var_name == "nondet_float" ||
        var_name == "nondet_bool")
      {
        // Generic nondet name - use argN format
        final_var_name = "arg" + std::to_string(current_param_names.size());
      }
      else
      {
        // Check if this name already exists in current_param_names
        int count = 0;
        for (const auto &existing_name : current_param_names)
        {
          if (
            existing_name == final_var_name ||
            existing_name.find(final_var_name) == 0)
            count++;
        }

        if (count > 0)
        {
          // Add numeric suffix to make it unique
          final_var_name = final_var_name + std::to_string(count);
        }
      }

      current_params.push_back(value_str);
      current_param_names.push_back(final_var_name);
    }
  }

  // Determine if we're building a dict (has dict components)
  // If so, skip list building because list_sizes are internal to dict construction
  bool has_dict_components = !dict_keys.empty() || !dict_values.empty();

  // Build composite list values from collected size and element values
  // Skip if we have dict components - the sizes are for the dict, not a separate list
  if (!has_dict_components)
  {
    if (list_elems.size() > list_sizes.size())
    {
      // Multi-element list: each loop iteration generated a fresh nondet value
      // Build a single list from all collected element values
      std::string list_str = "[";
      for (size_t j = 0; j < list_elems.size(); ++j)
      {
        if (j > 0)
          list_str += ", ";
        list_str += list_elems[j].second;
      }
      list_str += "]";
      current_params.push_back(list_str);
      current_param_names.push_back("list0");
    }
    else
    {
      // Passed elem_type case: each (size, elem) pair creates [elem] * size
      for (size_t i = 0; i < list_sizes.size() && i < list_elems.size(); ++i)
      {
        BigInt size = list_sizes[i].second;
        const std::string &elem = list_elems[i].second;

        std::string list_str = "[";
        int64_t size_val = size.to_int64();
        if (size_val < 0)
          size_val = 0;
        if (size_val > 100)
          size_val = 100; // Cap at reasonable size

        for (int64_t j = 0; j < size_val; ++j)
        {
          if (j > 0)
            list_str += ", ";
          list_str += elem;
        }
        list_str += "]";

        std::string param_name = "list" + std::to_string(i);
        current_params.push_back(list_str);
        current_param_names.push_back(param_name);
      }

      // Handle orphan list sizes (build list with default element 0)
      for (size_t i = list_elems.size(); i < list_sizes.size(); ++i)
      {
        BigInt size = list_sizes[i].second;
        int64_t size_val = size.to_int64();
        if (size_val < 0)
          size_val = 0;
        if (size_val > 100)
          size_val = 100;

        std::string list_str = "[";
        for (int64_t j = 0; j < size_val; ++j)
        {
          if (j > 0)
            list_str += ", ";
          list_str += "0";
        }
        list_str += "]";

        current_params.push_back(list_str);
        current_param_names.push_back("list" + std::to_string(i));
      }
    }
  }

  // Build composite dict values from collected key and value values
  if (!dict_keys.empty() || !dict_values.empty())
  {
    std::string dict_str = "{";
    size_t num_entries = std::max(dict_keys.size(), dict_values.size());

    for (size_t i = 0; i < num_entries; ++i)
    {
      if (i > 0)
        dict_str += ", ";

      std::string key_val;
      if (i < dict_keys.size())
        key_val = dict_keys[i].second;
      else
        key_val = std::to_string(i);

      std::string value_val;
      if (i < dict_values.size())
        value_val = dict_values[i].second;
      else if (i < dict_keys.size())
        value_val = dict_keys[i].second;
      else
        value_val = "0";

      dict_str += key_val + ": " + value_val;
    }
    dict_str += "}";

    current_params.push_back(dict_str);
    current_param_names.push_back("dict0");
  }

  // Store collected data if we found any nondet values
  if (!current_params.empty())
  {
    std::lock_guard<std::mutex> lock(data_mutex);

    // Build a map from parameter name to value
    std::map<std::string, std::string> param_map;
    for (size_t i = 0; i < current_param_names.size(); ++i)
      param_map[current_param_names[i]] = current_params[i];

    // Check if we have new parameters not seen before
    std::vector<std::string> new_param_names;
    for (const auto &name : current_param_names)
    {
      if (
        std::find(param_names.begin(), param_names.end(), name) ==
        param_names.end())
      {
        new_param_names.push_back(name);
      }
    }

    // If we found new parameters, update existing test cases with default values
    if (!new_param_names.empty())
    {
      for (const auto &new_name : new_param_names)
      {
        param_names.push_back(new_name);
        // Add default value to all existing test cases
        for (auto &test_case : test_cases)
          test_case.push_back("0");
      }
    }

    // Initialize param names on first collection
    if (param_names.empty())
    {
      param_names = current_param_names;
    }

    // Match parameters by name to handle condition-coverage mode where
    // different branches may collect variables in different orders or subsets
    std::vector<std::string> matched_params;
    matched_params.resize(param_names.size());

    // Match against canonical parameter list
    for (size_t i = 0; i < param_names.size(); ++i)
    {
      auto it = param_map.find(param_names[i]);
      if (it != param_map.end())
      {
        matched_params[i] = it->second;
      }
      else
      {
        // Parameter not found in this counterexample - use default value
        // This can happen in condition-coverage when a parameter isn't used in a branch
        matched_params[i] = "0";
      }
    }

    test_cases.push_back(matched_params);

    // Store function name if we found one
    if (!extracted_func_name.empty() && function_name.empty())
      function_name = extracted_func_name;
  }
}

void pytest_generator::generate(const std::string &file_name) const
{
  std::lock_guard<std::mutex> lock(data_mutex);

  if (test_cases.empty())
  {
    log_warning("No test cases collected. No pytest file generated.");
    return;
  }

  // Extract module name from input file
  std::string input_file = config.options.get_option("input-file");
  std::string module_name = extract_module_name(input_file);

  // Generate pytest file
  std::ofstream pytest_file(file_name);

  // Write file components using helper methods
  write_file_header(pytest_file, input_file);
  write_imports(pytest_file, module_name);
  write_test_data(pytest_file, param_names, test_cases);

  // Generate test function
  std::string test_func_name =
    function_name.empty() ? "coverage" : function_name;
  write_test_function(pytest_file, test_func_name, param_names);

  pytest_file.close();
  log_status(
    "Generated pytest test case with {} test(s): {}",
    test_cases.size(),
    file_name);
}

bool pytest_generator::has_tests() const
{
  std::lock_guard<std::mutex> lock(data_mutex);
  return !test_cases.empty();
}

void pytest_generator::generate_single(
  const std::string &file_name,
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  const namespacet &ns)
{
  (void)ns; // Suppress unused parameter warning

  // Extract original Python file name and module name
  std::string original_file = config.options.get_option("input-file");
  std::string module_name = extract_module_name(original_file);

  // Track nondet symbols we've seen to avoid duplicates
  std::unordered_set<std::string> seen_nondets;

  // Current test case parameters (in order)
  std::vector<std::string> current_params;
  std::vector<std::string> current_param_names;

  // Track nondet_list/nondet_dict components for building composite values
  std::vector<std::pair<std::string, BigInt>> list_sizes;
  std::vector<std::pair<std::string, std::string>> list_elems;
  std::vector<std::pair<std::string, std::string>> dict_keys;
  std::vector<std::pair<std::string, std::string>> dict_values;

  // Extract function name
  std::string func_name = extract_function_name(target, smt_conv);
  if (func_name.empty())
    func_name = "test_function"; // Fallback

  // Traverse SSA steps to extract nondet variables
  for (auto const &SSA_step : target.SSA_steps)
  {
    if (!smt_conv.l_get(SSA_step.guard_ast).is_true())
      continue;

    if (SSA_step.is_assignment())
    {
      // Extract the variable name from lhs first
      std::string var_name;
      if (is_symbol2t(SSA_step.lhs))
      {
        const symbol2t &lhs_sym = to_symbol2t(SSA_step.lhs);
        var_name = clean_variable_name(lhs_sym.get_symbol_name());
      }

      // Track nondet_list/nondet_dict internal variables to build composite values
      bool is_list_size = false;
      bool is_list_elem = false;
      bool is_dict_key = false;
      bool is_dict_value = false;

      if (SSA_step.source.pc->location.function() != "")
      {
        std::string inner_func_name =
          SSA_step.source.pc->location.function().as_string();

        // Check if we're inside an internal function
        bool in_nondet_size = false;
        bool in_nondet_list = false;
        bool in_nondet_dict = false;

        if (inner_func_name == "_nondet_size")
          in_nondet_size = true;
        else if (inner_func_name == "nondet_list")
          in_nondet_list = true;
        else if (inner_func_name == "nondet_dict")
          in_nondet_dict = true;
        else
        {
          // Check for mangled names
          size_t sep_pos = inner_func_name.rfind("::");
          if (sep_pos != std::string::npos)
          {
            std::string base_name = inner_func_name.substr(sep_pos + 2);
            if (base_name == "_nondet_size")
              in_nondet_size = true;
            else if (base_name == "nondet_list")
              in_nondet_list = true;
            else if (base_name == "nondet_dict")
              in_nondet_dict = true;
          }
        }

        // Track the role of internal variables
        if (in_nondet_size && var_name == "size")
        {
          is_list_size = true;
        }
        else if (in_nondet_list)
        {
          if (var_name == "elem_type" || var_name == "elem")
            is_list_elem = true;
        }
        else if (in_nondet_dict)
        {
          if (var_name == "key_type" || var_name == "k")
            is_dict_key = true;
          else if (var_name == "value_type" || var_name == "v")
            is_dict_value = true;
        }

        // Skip only loop counters and result variables
        if (
          (in_nondet_list || in_nondet_dict || in_nondet_size) &&
          (var_name == "i" || var_name == "result"))
        {
          continue;
        }
      }

      // Also check variable names outside internal functions
      // This handles cases where user passes nondet_int() as argument
      if (!is_list_size && !is_list_elem && !is_dict_key && !is_dict_value)
      {
        if (var_name == "key_type")
          is_dict_key = true;
        else if (var_name == "value_type")
          is_dict_value = true;
        else if (var_name == "elem_type")
          is_list_elem = true;
      }

      // Check if this is a nondet assignment
      auto nondet_expr = symex_slicet::get_nondet_symbol(SSA_step.rhs);
      if (!nondet_expr || !is_symbol2t(nondet_expr))
        continue;

      const symbol2t &sym = to_symbol2t(nondet_expr);
      if (!has_prefix(sym.thename.as_string(), "nondet$"))
        continue;

      // For dict/list components, allow duplicate nondet symbols
      // (key_type and value_type may share the same nondet symbol due to solver optimization)
      bool is_component = is_list_size || is_list_elem || is_dict_key || is_dict_value;

      if (seen_nondets.count(sym.thename.as_string()) && !is_component)
        continue;

      if (!seen_nondets.count(sym.thename.as_string()))
        seen_nondets.insert(sym.thename.as_string());

      // Get the concrete value from the solver
      auto concrete_value = smt_conv.get(nondet_expr);

      std::string value_str;
      if (is_constant_int2t(concrete_value))
      {
        BigInt int_value = to_constant_int2t(concrete_value).value;
        value_str = integer2string(int_value);

        // If this is a list/dict size, store it for later use
        if (is_list_size)
        {
          list_sizes.push_back({sym.thename.as_string(), int_value});
          continue;
        }
      }
      else if (is_constant_floatbv2t(concrete_value))
      {
        std::string c_float =
          to_constant_floatbv2t(concrete_value).value.to_ansi_c_string();
        value_str = convert_float_to_python(c_float);
      }
      else if (is_constant_bool2t(concrete_value))
        value_str = to_constant_bool2t(concrete_value).value ? "True" : "False";
      else if (is_constant_string2t(concrete_value))
      {
        std::string raw_str =
          to_constant_string2t(concrete_value).value.as_string();
        value_str = "\"" + escape_python_string(raw_str) + "\"";
      }
      else if (is_constant_array2t(concrete_value))
      {
        // Check if this is a character array (string) first
        if (is_char_array(concrete_value))
        {
          // Python string from char array (nondet_str)
          value_str = convert_char_array_to_string(concrete_value);
        }
        else
        {
          // Python list (nondet_list)
          value_str = convert_array_to_python_list(concrete_value);
        }
      }
      else if (is_constant_struct2t(concrete_value))
      {
        // Python dict (nondet_dict)
        value_str = convert_struct_to_python_dict(concrete_value);
      }
      else
        value_str = "None"; // Unsupported type

      // Store list/dict element values for later combination
      if (is_list_elem)
      {
        list_elems.push_back({sym.thename.as_string(), value_str});
        continue;
      }
      if (is_dict_key)
      {
        dict_keys.push_back({sym.thename.as_string(), value_str});
        continue;
      }
      if (is_dict_value)
      {
        dict_values.push_back({sym.thename.as_string(), value_str});
        continue;
      }

      // Handle duplicate parameter names by adding numeric suffix
      std::string final_var_name = var_name;
      if (
        var_name.empty() || var_name == "nondet_str" ||
        var_name == "nondet_list" || var_name == "nondet_dict" ||
        var_name == "nondet_int" || var_name == "nondet_float" ||
        var_name == "nondet_bool")
      {
        // Generic nondet name - use argN format
        final_var_name = "arg" + std::to_string(current_param_names.size());
      }
      else
      {
        // Check if this name already exists in current_param_names
        int count = 0;
        for (const auto &existing_name : current_param_names)
        {
          if (
            existing_name == final_var_name ||
            existing_name.find(final_var_name) == 0)
            count++;
        }

        if (count > 0)
        {
          // Add numeric suffix to make it unique
          final_var_name = final_var_name + std::to_string(count);
        }
      }

      current_params.push_back(value_str);
      current_param_names.push_back(final_var_name);
    }
  }

  // Determine if we're building a dict (has dict components)
  // If so, skip list building because list_sizes are internal to dict construction
  bool has_dict_components = !dict_keys.empty() || !dict_values.empty();

  // Build composite list values from collected size and element values
  // Skip if we have dict components - the sizes are for the dict, not a separate list
  if (!has_dict_components)
  {
    if (list_elems.size() > list_sizes.size())
    {
      // Multi-element list: each loop iteration generated a fresh nondet value
      // Build a single list from all collected element values
      std::string list_str = "[";
      for (size_t j = 0; j < list_elems.size(); ++j)
      {
        if (j > 0)
          list_str += ", ";
        list_str += list_elems[j].second;
      }
      list_str += "]";
      current_params.push_back(list_str);
      current_param_names.push_back("list0");
    }
    else
    {
      // Passed elem_type case: each (size, elem) pair creates [elem] * size
      for (size_t i = 0; i < list_sizes.size() && i < list_elems.size(); ++i)
      {
        BigInt size = list_sizes[i].second;
        const std::string &elem = list_elems[i].second;

        std::string list_str = "[";
        int64_t size_val = size.to_int64();
        if (size_val < 0)
          size_val = 0;
        if (size_val > 100)
          size_val = 100;

        for (int64_t j = 0; j < size_val; ++j)
        {
          if (j > 0)
            list_str += ", ";
          list_str += elem;
        }
        list_str += "]";

        std::string param_name = "list" + std::to_string(i);
        current_params.push_back(list_str);
        current_param_names.push_back(param_name);
      }

      // Handle orphan list sizes (build list with default element 0)
      for (size_t i = list_elems.size(); i < list_sizes.size(); ++i)
      {
        BigInt size = list_sizes[i].second;
        int64_t size_val = size.to_int64();
        if (size_val < 0)
          size_val = 0;
        if (size_val > 100)
          size_val = 100;

        std::string list_str = "[";
        for (int64_t j = 0; j < size_val; ++j)
        {
          if (j > 0)
            list_str += ", ";
          list_str += "0";
        }
        list_str += "]";

        current_params.push_back(list_str);
        current_param_names.push_back("list" + std::to_string(i));
      }
    }
  }

  // Build composite dict values from collected key and value values
  if (!dict_keys.empty() || !dict_values.empty())
  {
    std::string dict_str = "{";
    size_t num_entries = std::max(dict_keys.size(), dict_values.size());

    for (size_t i = 0; i < num_entries; ++i)
    {
      if (i > 0)
        dict_str += ", ";

      std::string key_val;
      if (i < dict_keys.size())
        key_val = dict_keys[i].second;
      else
        key_val = std::to_string(i);

      std::string value_val;
      if (i < dict_values.size())
        value_val = dict_values[i].second;
      else if (i < dict_keys.size())
        value_val = dict_keys[i].second;
      else
        value_val = "0";

      dict_str += key_val + ": " + value_val;
    }
    dict_str += "}";

    current_params.push_back(dict_str);
    current_param_names.push_back("dict0");
  }

  // If no nondets found, nothing to generate
  if (current_params.empty())
  {
    log_warning("No nondet variables found. No pytest test case generated.");
    return;
  }

  // Generate pytest file
  std::ofstream pytest_file(file_name);

  // Write file components using helper methods
  write_file_header(pytest_file, original_file);
  write_imports(pytest_file, module_name);

  // Convert single test case to format expected by write_test_data
  std::vector<std::vector<std::string>> test_data = {current_params};
  write_test_data(pytest_file, current_param_names, test_data);
  write_test_function(pytest_file, func_name, current_param_names);

  pytest_file.close();
  log_status("Generated pytest test case: {}", file_name);
}