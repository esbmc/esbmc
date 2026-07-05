#include "python_list_internal.h"

using namespace python_expr;

exprt python_list::build_split_list(
  python_converter &converter,
  const nlohmann::json &call_node,
  const std::string &input,
  const std::string &separator,
  long long count,
  bool from_right)
{
  if (separator.empty())
  {
    // rsplit() with no separator (whitespace) and an explicit maxsplit has
    // subtle leading/trailing-whitespace asymmetry vs split() and is not
    // supported; reject it cleanly rather than risk a wrong result. Without a
    // maxsplit, rsplit(None) == split(None) (same tokens, same order), so a
    // negative count falls through to the shared whitespace logic below.
    if (from_right && count >= 0)
      throw std::runtime_error(
        "rsplit() with maxsplit and no separator is not supported");

    // Whitespace split: split on any whitespace and collapse runs.
    auto is_space = [](char c) {
      return std::isspace(static_cast<unsigned char>(c)) != 0;
    };

    if (count == 0)
    {
      size_t first = 0;
      while (first < input.size() && is_space(input[first]))
        ++first;

      if (first == input.size())
      {
        nlohmann::json list_node;
        list_node["_type"] = "List";
        list_node["elts"] = nlohmann::json::array();
        converter.copy_location_fields_from_decl(call_node, list_node);
        python_list list(converter, list_node);
        return list.get();
      }

      // Remainder kept verbatim: leading whitespace is skipped, but CPython
      // keeps trailing whitespace for split(None, 0)
      // (e.g. '  a  '.split(None, 0) == ['a  ']).
      nlohmann::json list_node;
      list_node["_type"] = "List";
      list_node["elts"] = nlohmann::json::array();
      converter.copy_location_fields_from_decl(call_node, list_node);

      nlohmann::json elem;
      elem["_type"] = "Constant";
      elem["value"] = input.substr(first);
      converter.copy_location_fields_from_decl(call_node, elem);
      list_node["elts"].push_back(elem);

      python_list list(converter, list_node);
      return list.get();
    }

    std::vector<std::string> parts;
    size_t i = 0;
    const size_t n = input.size();

    auto skip_ws = [&](size_t &idx) {
      while (idx < n && is_space(input[idx]))
        ++idx;
    };

    auto scan_token = [&](size_t &idx) {
      while (idx < n && !is_space(input[idx]))
        ++idx;
    };

    skip_ws(i);
    if (i == n)
    {
      nlohmann::json list_node;
      list_node["_type"] = "List";
      list_node["elts"] = nlohmann::json::array();
      converter.copy_location_fields_from_decl(call_node, list_node);
      python_list list(converter, list_node);
      return list.get();
    }

    long long remaining = count;
    while (i < n)
    {
      size_t start = i;
      scan_token(i);
      parts.push_back(input.substr(start, i - start));

      if (count >= 0 && remaining == 1)
      {
        skip_ws(i);
        if (i < n)
          parts.push_back(input.substr(i));
        break;
      }

      if (count >= 0)
        --remaining;

      skip_ws(i);
      if (i >= n)
        break;
    }

    nlohmann::json list_node;
    list_node["_type"] = "List";
    list_node["elts"] = nlohmann::json::array();
    converter.copy_location_fields_from_decl(call_node, list_node);

    for (const auto &part : parts)
    {
      nlohmann::json elem;
      elem["_type"] = "Constant";
      elem["value"] = part;
      converter.copy_location_fields_from_decl(call_node, elem);
      list_node["elts"].push_back(elem);
    }

    python_list list(converter, list_node);
    return list.get();
  }

  if (count == 0)
  {
    nlohmann::json list_node;
    list_node["_type"] = "List";
    list_node["elts"] = nlohmann::json::array();
    converter.copy_location_fields_from_decl(call_node, list_node);

    nlohmann::json elem;
    elem["_type"] = "Constant";
    elem["value"] = input;
    converter.copy_location_fields_from_decl(call_node, elem);
    list_node["elts"].push_back(elem);

    python_list list(converter, list_node);
    return list.get();
  }

  std::vector<std::string> parts;
  if (from_right && count >= 1)
  {
    // rsplit(sep, count): keep the rightmost `count` splits. Compute the full
    // split, then merge the surplus leftmost parts back together with the
    // separator — this yields exactly Python's rsplit result and avoids
    // backward-scan boundary fiddliness. (count == 0 returned [input] above;
    // count < 0 is the unlimited case, identical to split, handled below.)
    std::vector<std::string> all;
    size_t s = 0;
    while (true)
    {
      size_t pos = input.find(separator, s);
      if (pos == std::string::npos)
      {
        all.push_back(input.substr(s));
        break;
      }
      all.push_back(input.substr(s, pos - s));
      s = pos + separator.size();
    }

    const long long total_splits = static_cast<long long>(all.size()) - 1;
    if (total_splits <= count)
      parts = all;
    else
    {
      const size_t merge_upto = all.size() - static_cast<size_t>(count);
      std::string merged = all[0];
      for (size_t k = 1; k < merge_upto; ++k)
        merged += separator + all[k];
      parts.push_back(merged);
      for (size_t k = merge_upto; k < all.size(); ++k)
        parts.push_back(all[k]);
    }
  }
  else
  {
    size_t start = 0;
    long long splits = 0;
    while (true)
    {
      if (count >= 0 && splits >= count)
      {
        parts.push_back(input.substr(start));
        break;
      }

      size_t pos = input.find(separator, start);
      if (pos == std::string::npos)
      {
        parts.push_back(input.substr(start));
        break;
      }
      parts.push_back(input.substr(start, pos - start));
      start = pos + separator.size();
      ++splits;
    }
  }

  nlohmann::json list_node;
  list_node["_type"] = "List";
  list_node["elts"] = nlohmann::json::array();
  converter.copy_location_fields_from_decl(call_node, list_node);

  for (const auto &part : parts)
  {
    nlohmann::json elem;
    elem["_type"] = "Constant";
    elem["value"] = part;
    converter.copy_location_fields_from_decl(call_node, elem);
    list_node["elts"].push_back(elem);
  }

  python_list list(converter, list_node);
  return list.get();
}

exprt python_list::build_split_list(
  python_converter &converter,
  const nlohmann::json &call_node,
  const exprt &input_expr,
  const std::string &separator,
  long long count,
  bool from_right)
{
  // For symbolic strings, we create a runtime call to __python_str_split
  // This function will handle the splitting at runtime with symbolic constraints

  // The runtime model splits left-to-right, so it only models rsplit() when no
  // maxsplit limits the result (rsplit() == split() then). A right-anchored
  // maxsplit on a non-constant string would need a dedicated model; reject it
  // cleanly rather than return a wrong result.
  if (from_right && count >= 0)
    throw std::runtime_error(
      "rsplit() with maxsplit on a non-constant string is not supported");

  locationt location = converter.get_location_from_decl(call_node);

  // Create function symbol for __python_str_split if it doesn't exist
  const std::string func_name = "c:@F@__python_str_split";
  const symbolt *func_symbol = converter.symbol_table().find_symbol(func_name);

  if (!func_symbol)
  {
    // Create function type: PyListObject* __python_str_split(char* str, char* sep, int maxsplit)
    code_typet func_type;
    func_type.return_type() = converter.get_type_handler().get_list_type();

    code_typet::argumentt str_arg;
    str_arg.type() = pointer_typet(char_type());
    func_type.arguments().push_back(str_arg);

    code_typet::argumentt sep_arg;
    sep_arg.type() = pointer_typet(char_type());
    func_type.arguments().push_back(sep_arg);

    code_typet::argumentt count_arg;
    count_arg.type() = long_long_int_type();
    func_type.arguments().push_back(count_arg);

    symbolt new_symbol;
    new_symbol.name = func_name;
    new_symbol.id = func_name;
    new_symbol.set_type(func_type);
    new_symbol.mode = "C";
    new_symbol.module = "python";
    new_symbol.location = location;
    new_symbol.is_extern = true;

    converter.add_symbol(new_symbol);
    func_symbol = converter.symbol_table().find_symbol(func_name);
  }

  // Build arguments for the call
  exprt::operandst args;

  // Argument 1: input string (ensure it's a pointer)
  exprt str_arg = input_expr;
  if (str_arg.type().is_array())
  {
    // Get address of first element
    str_arg = converter.get_string_handler().get_array_base_address(str_arg);
  }
  args.push_back(str_arg);

  // Argument 2: separator string
  std::string sep_to_use = separator.empty() ? "" : separator;
  exprt sep_expr =
    converter.get_string_builder().build_string_literal(sep_to_use);
  if (sep_expr.type().is_array())
  {
    sep_expr = converter.get_string_handler().get_array_base_address(sep_expr);
  }
  args.push_back(sep_expr);

  // Argument 3: maxsplit count
  exprt count_expr = from_integer(count, long_long_int_type());
  args.push_back(count_expr);

  // Create a temp list symbol to hold the split result.
  const typet list_type = converter.get_type_handler().get_list_type();
  symbolt &split_list =
    converter.create_tmp_symbol(call_node, "$split_list$", list_type, exprt());
  code_declt split_decl(build_symbol(split_list));
  split_decl.location() = location;
  converter.add_instruction(split_decl);

  // Emit the function call with lhs so the list has a stable identifier.
  code_function_callt split_call;
  split_call.function() = build_symbol(*func_symbol);
  split_call.arguments() = args;
  split_call.lhs() = build_symbol(split_list);
  split_call.type() = list_type;
  split_call.location() = location;
  converter.add_instruction(split_call);

  // Record element type as string to ensure correct comparisons on parts[i].
  typet elem_type = converter.get_type_handler().build_array(char_type(), 0);
  list_type_map[split_list.id.as_string()].push_back(
    std::make_pair(std::string(), elem_type));

  return build_symbol(split_list);
}
