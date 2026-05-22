#include <python-frontend/json_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/symbol_id.h>
#include <util/arith_tools.h>
#include <util/message.h>

#include <regex>

void python_converter::update_symbol(const exprt &expr) const
{
  // Don't update if expression has no name
  // prevents corruption of function symbols
  if (expr.name().empty())
  {
    log_debug(
      "python-frontend",
      "[update_symbol]: skipping symbol update since expression has no name");
    return;
  }

  // Generate a symbol ID from the expression's name.
  symbol_id sid = create_symbol_id();
  sid.set_object(expr.name().c_str());

  // Try to locate the symbol in the symbol table.
  symbolt *sym = symbol_table_.find_symbol(sid.to_string());

  if (sym == nullptr)
  {
    // Symbol not found, nothing to update.
    return;
  }

  // Update the type of the symbol and its value.
  const typet &expr_type = expr.type();
  sym->get_type() = expr_type;
  sym->get_value().type() = expr_type;

  // Check if the symbol has a constant or bitvector value.
  if (
    sym->get_value().is_constant() || sym->get_value().is_signedbv() ||
    sym->get_value().is_unsignedbv())
  {
    const std::string &binary_value_str = sym->get_value().value().c_str();

    // Only attempt binary conversion if the string is non-empty and consists
    // solely of '0' and '1' characters (i.e., it is a valid binary string).
    // Character or decimal values stored in the symbol will not satisfy this
    // check and must be left unchanged to avoid a stoll conversion failure.
    bool is_binary_string =
      !binary_value_str.empty() &&
      binary_value_str.find_first_not_of("01") == std::string::npos;

    if (is_binary_string)
    {
      try
      {
        // Convert binary string to integer. binary2integer accepts any width,
        // so we don't lose precision on bignum constants emitted by the
        // --ir-gated ** widening for #1964 Part 2 / #4642.
        const bool is_signed = expr_type.is_signedbv();
        const BigInt int_val = binary2integer(binary_value_str, is_signed);

        // Create a new constant expression with the converted value and type.
        exprt new_value = from_integer(int_val, expr_type);

        // Assign the new value to the symbol.
        sym->get_value() = new_value;
      }
      catch (const std::exception &e)
      {
        log_error(
          "update_symbol: Failed to convert binary value '{}' to integer for "
          "symbol '{}'. Error: {}",
          binary_value_str,
          sid.to_string(),
          e.what());
      }
    }
  }
}

symbolt *python_converter::find_function_in_base_classes(
  const std::string &class_name,
  const std::string &symbol_id,
  std::string method_name,
  bool is_ctor) const
{
  symbolt *func = nullptr;

  // Find class node in the AST
  auto class_node = json_utils::find_class((*ast_json)["body"], class_name);

  if (class_node != nlohmann::json())
  {
    std::string current_class = class_name;
    std::string current_func_name = (is_ctor) ? class_name : method_name;
    std::string sym_id = symbol_id;
    // Search for method in all bases classes
    for (const auto &base_class_node : class_node["bases"])
    {
      const std::string &base_class = base_class_node["id"].get<std::string>();
      if (is_ctor)
        method_name = base_class;

      std::size_t pos = sym_id.rfind("@C@" + current_class);

      sym_id.replace(
        pos,
        std::string("@C@" + current_class + "@F@" + current_func_name).length(),
        std::string("@C@" + base_class + "@F@" + method_name));

      if ((func = symbol_table_.find_symbol(sym_id.c_str())))
        return func;

      current_class = base_class;
    }
  }

  return func;
}

symbolt *
python_converter::find_imported_symbol(const std::string &symbol_id) const
{
  // Extract the name being looked up from the symbol ID.
  // When the symbol has a class component (py:main@C@Foo@F@bar),
  // use the class name for matching against import names.
  auto parsed = ::symbol_id::from_string(symbol_id);
  std::string lookup_name =
    !parsed.get_class().empty()
      ? parsed.get_class()
      : (parsed.get_function().empty() ? parsed.get_object()
                                       : parsed.get_function());

  // symbol_id::from_string currently parses class/function components but not
  // trailing object names (e.g. py:file@replace). Recover that case from raw
  // text so imported free functions are still resolved.
  if (lookup_name.empty())
  {
    const std::size_t at = symbol_id.rfind('@');
    if (at != std::string::npos && at + 1 < symbol_id.size())
    {
      lookup_name = symbol_id.substr(at + 1);
      if (lookup_name == "C" || lookup_name == "F")
        lookup_name.clear();
    }
  }

  auto find_in_import = [&](const nlohmann::json &obj) -> symbolt * {
    if (
      (obj["_type"] == "ImportFrom" || obj["_type"] == "Import") &&
      obj.contains("full_path") && !obj["full_path"].is_null())
    {
      // For ImportFrom, only match if the specific name was imported.
      // This prevents "from other import sum" from also hijacking "max".
      if (
        obj["_type"] == "ImportFrom" && obj.contains("names") &&
        !lookup_name.empty())
      {
        bool name_imported = false;
        for (const auto &name : obj["names"])
        {
          const std::string &n = name["name"].get<std::string>();
          if (n == "*" || n == lookup_name)
          {
            name_imported = true;
            break;
          }
          if (
            name.contains("asname") && !name["asname"].is_null() &&
            name["asname"].get<std::string>() == lookup_name)
          {
            name_imported = true;
            break;
          }
        }
        if (!name_imported)
          return nullptr;
      }

      std::regex pattern("py:(.*?)@");
      std::string imported_symbol = std::regex_replace(
        symbol_id, pattern, "py:" + obj["full_path"].get<std::string>() + "@");

      if (
        symbolt *func_symbol =
          symbol_table_.find_symbol(imported_symbol.c_str()))
        return func_symbol;

      // Imported free functions are often looked up as object symbols in the
      // caller scope (e.g., py:main@replace). Also probe the equivalent
      // function-id form in the imported module (py:module@F@replace).
      if (!lookup_name.empty())
      {
        ::symbol_id imported_sid = ::symbol_id::from_string(imported_symbol);
        imported_sid.set_class("");
        imported_sid.set_object("");
        imported_sid.set_attribute("");
        imported_sid.set_function(lookup_name);

        if (
          symbolt *func_symbol =
            symbol_table_.find_symbol(imported_sid.to_string().c_str()))
          return func_symbol;

        imported_sid.set_function("");
        imported_sid.set_object(lookup_name);
        if (
          symbolt *obj_symbol =
            symbol_table_.find_symbol(imported_sid.to_string().c_str()))
          return obj_symbol;
      }
    }

    return nullptr;
  };

  for (const auto &obj : (*ast_json)["body"])
  {
    if (symbolt *func_symbol = find_in_import(obj))
      return func_symbol;
  }

  // Also check imports inside function bodies.
  for (const auto &obj : (*ast_json)["body"])
  {
    if (
      obj["_type"] == "FunctionDef" && obj.contains("body") &&
      obj["body"].is_array())
    {
      for (const auto &stmt : obj["body"])
      {
        if (symbolt *func_symbol = find_in_import(stmt))
          return func_symbol;
      }
    }
  }

  return nullptr;
}

symbolt *
python_converter::find_nested_function_symbol(const std::string &name) const
{
  if (name.empty() || current_func_name_.empty())
    return nullptr;

  for (std::string scope = current_func_name_; !scope.empty();)
  {
    ::symbol_id nested_func_sid(current_python_file, "", scope + "@F@" + name);
    if (
      symbolt *nested_func_symbol =
        symbol_table_.find_symbol(nested_func_sid.to_string()))
    {
      if (nested_func_symbol->get_type().is_code())
        return nested_func_symbol;
    }

    const std::size_t sep = scope.rfind("@F@");
    if (sep == std::string::npos)
      break;
    scope.resize(sep);
  }

  return nullptr;
}

symbolt *python_converter::find_symbol(const std::string &sym_id) const
{
  // When not loading models, check imports first so that user imports
  // (e.g. "from other import sum") shadow builtin model functions that
  // are registered in the main file's namespace.
  // Don't let model-module stubs (e.g. esbmc.py's nondet_list
  // stub) shadow operational models (e.g. nondet.py's nondet_list).
  if (!is_loading_models)
  {
    if (symbolt *imported = find_imported_symbol(sym_id))
    {
      const std::string &imp_id = imported->id.as_string();
      bool is_model_stub = imp_id.find("/models/") != std::string::npos;
      if (!is_model_stub || !symbol_table_.find_symbol(sym_id))
        return imported;
    }
  }

  if (symbolt *symbol = symbol_table_.find_symbol(sym_id))
    return symbol;

  symbol_id nested_sym_id = symbol_id::from_string(sym_id);
  std::size_t pos = sym_id.rfind('@');
  std::string obj = (pos == std::string::npos) ? "" : sym_id.substr(pos + 1);
  nested_sym_id.set_object(obj);

  for (int i = scope_stack_.size(); i >= 0; --i)
  {
    // Build id prefix from module + outer scopes
    for (int j = 0; j < i; ++j)
    {
      std::string func = scope_stack_[j].substr(3); // drop "@F@"
      nested_sym_id.set_function(func);
      std::string candidate = nested_sym_id.to_string();

      if (symbolt *symbol = symbol_table_.find_symbol(candidate))
        return symbol;
    }
  }

  if (symbolt *symbol = find_symbol_in_global_scope(sym_id))
    return symbol;
  return find_imported_symbol(sym_id);
}

symbolt *python_converter::find_symbol_in_global_scope(
  const std::string &symbol_id) const
{
  std::size_t class_start_pos = symbol_id.find("@C@");
  std::size_t func_start_pos = symbol_id.find("@F@");
  std::string sid = symbol_id;

  // Remove class name from symbol
  if (class_start_pos != std::string::npos)
    sid.erase(class_start_pos, func_start_pos - class_start_pos);

  func_start_pos = sid.find("@F@");
  std::size_t func_end_pos = sid.rfind("@");

  // Remove function name from symbol
  if (func_start_pos != std::string::npos)
    sid.erase(func_start_pos, func_end_pos - func_start_pos);

  return symbol_table_.find_symbol(sid);
}

bool python_converter::is_imported_module(const std::string &module_name) const
{
  if (imported_modules.find(module_name) != imported_modules.end())
    return true;

  return json_utils::is_module(module_name, *ast_json);
}

symbolt &python_converter::create_tmp_symbol(
  const nlohmann::json &element,
  const std::string var_name,
  const typet &symbol_type,
  const exprt &symbol_value)
{
  locationt location = get_location_from_decl(element);
  std::string path = location.file().as_string();
  std::string name_prefix =
    path + ":" + location.get_line().as_string() + var_name;
  symbolt &cl =
    sym_generator_.new_symbol(symbol_table_, symbol_type, name_prefix);
  cl.mode = "Python";
  std::string module_name = location.get_file().as_string();
  cl.module = module_name;
  cl.location = location;
  cl.static_lifetime = false;
  cl.is_extern = false;
  cl.file_local = true;
  if (symbol_value != exprt())
    cl.get_value() = symbol_value;

  return cl;
}
