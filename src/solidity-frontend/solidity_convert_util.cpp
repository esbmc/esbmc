/// \file solidity_convert_util.cpp
/// \brief Utility and helper functions for the Solidity converter.
///
/// Provides shared utility methods used across the converter: source location
/// extraction, AST node search by ID, parent node lookup, contract name
/// resolution, line number computation from source ranges, JSON AST traversal
/// helpers, and various name/ID construction routines.

#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_expr.h>
#include <util/message.h>
#include <regex>
#include <optional>

#include <fstream>

void solidity_convertert::get_location_from_node(
  const nlohmann::json &ast_node,
  locationt &location)
{
  location.set_line(get_line_number(ast_node));
  location.set_file(
    absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory

  // To annotate local declaration within a function
  if (current_functionDecl)
  {
    location.set_function(
      current_functionName); // set the function where this local variable belongs to
  }
}

void solidity_convertert::get_start_location_from_stmt(
  const nlohmann::json &ast_node,
  locationt &location)
{
  std::string function_name;

  if (current_functionDecl)
    function_name = current_functionName;

  // The src manager of Solidity AST JSON is too encryptic.
  // For the time being we are setting it to "1".
  location.set_line(get_line_number(ast_node));
  location.set_file(
    absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory

  if (!function_name.empty())
    location.set_function(function_name);
}

void solidity_convertert::get_final_location_from_stmt(
  const nlohmann::json &ast_node,
  locationt &location)
{
  std::string function_name;

  if (current_functionDecl)
    function_name = current_functionName;

  // The src manager of Solidity AST JSON is too encryptic.
  // For the time being we are setting it to "1".
  location.set_line(get_line_number(ast_node, true));
  location.set_file(
    absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory

  if (!function_name.empty())
    location.set_function(function_name);
}

unsigned int solidity_convertert::get_line_number(
  const nlohmann::json &ast_node,
  bool final_position)
{
  // Solidity src means "start:length:index", where "start" represents the position of the first char byte of the identifier.
  std::string src = ast_node.contains("src")
                      ? ast_node["src"].get<std::string>()
                      : get_src_from_json(ast_node);

  std::string position = src.substr(0, src.find(":"));
  unsigned int byte_position = std::stoul(position) + 1;

  if (final_position)
    byte_position = add_offset(src, byte_position);

  // the line number can be calculated by counting the number of line breaks prior to the identifier.
  unsigned int loc = std::count(
                       contract_contents.begin(),
                       (contract_contents.begin() + byte_position),
                       '\n') +
                     1;
  return loc;
}

unsigned int solidity_convertert::add_offset(
  const std::string &src,
  unsigned int start_position)
{
  // extract the length from "start:length:index"
  std::string offset = src.substr(1, src.find(":"));
  // already added 1 in start_position
  unsigned int end_position = start_position + std::stoul(offset);
  return end_position;
}

std::string
solidity_convertert::get_src_from_json(const nlohmann::json &ast_node)
{
  // some nodes may have "src" inside a member json object
  // we need to deal with them case by case based on the node type
  SolidityGrammar::ExpressionT type =
    SolidityGrammar::get_expression_t(ast_node);
  switch (type)
  {
  case SolidityGrammar::ExpressionT::ImplicitCastExprClass:
  {
    assert(ast_node.contains("subExpr"));
    assert(ast_node["subExpr"].contains("src"));
    return ast_node["subExpr"]["src"].get<std::string>();
  }
  case SolidityGrammar::ExpressionT::NullExpr:
  {
    // empty address
    return "-1:-1:-1";
  }
  default:
  {
    log_error("Unsupported node type when getting src from JSON");
    abort();
  }
  }
}

std::string solidity_convertert::get_modulename_from_path(std::string path)
{
  std::string filename = get_filename_from_path(path);

  if (filename.find_last_of('.') != std::string::npos)
    return filename.substr(0, filename.find_last_of('.'));

  return filename;
}

std::string solidity_convertert::get_filename_from_path(std::string path)
{
  if (path.find_last_of('/') != std::string::npos)
    return path.substr(path.find_last_of('/') + 1);

  return path; // for _x, it just returns "overflow_2.c" because the test program is in the same dir as esbmc binary
}

bool solidity_convertert::get_constant_value(
  const int ref_id,
  std::string &value)
{
  log_debug("solidity", "get constant var's value");
  nlohmann::json tmp = find_node_by_id(src_ast_json, ref_id);
  while (!tmp.empty() && tmp.contains("value"))
  {
    auto val_json = tmp["value"];
    if (!val_json.contains("value"))
    {
      assert(val_json.contains("referencedDeclaration"));
      int new_ref_id = val_json["referencedDeclaration"].get<int>();
      tmp = find_node_by_id(src_ast_json, new_ref_id);
    }
    else
    {
      value = tmp["value"]["value"].get<std::string>();
      return false;
    }
  }

  return true;
}

void solidity_convertert::get_default_symbol(
  symbolt &symbol,
  std::string module_name,
  typet type,
  std::string name,
  std::string id,
  locationt location)
{
  symbol.mode = mode;
  symbol.module = module_name;
  symbol.location = std::move(location);
  symbol.type = std::move(type);
  symbol.name = name;
  symbol.id = id;
}

symbolt *solidity_convertert::move_symbol_to_context(symbolt &symbol)
{
  return context.move_symbol_to_context(symbol);
}

void solidity_convertert::convert_expression_to_code(exprt &expr)
{
  if (expr.is_code())
    return;

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  expr.swap(code);
}

bool solidity_convertert::check_intrinsic_function(
  const nlohmann::json &ast_node)
{
  // function to detect special intrinsic functions, e.g. ___ESBMC_assume
  return (
    ast_node.contains("name") && (ast_node["name"] == "__ESBMC_assume" ||
                                  ast_node["name"] == "__VERIFIER_assume" ||
                                  ast_node["name"] == "__ESBMC_assert" ||
                                  ast_node["name"] == "__VERIFIER_assert"));
}

nlohmann::json solidity_convertert::make_implicit_cast_expr(
  const nlohmann::json &sub_expr,
  std::string cast_type)
{
  log_debug("solidity", "\t@@@ make_implicit_cast_expr");
  // Since Solidity AST does not have type cast information about return values,
  // we need to manually make a JSON object and wrap the return expression in it.
  std::map<std::string, std::string> m = {
    {"nodeType", "ImplicitCastExprClass"},
    {"castType", cast_type},
    {"subExpr", {}}};
  nlohmann::json implicit_cast_expr = m;
  implicit_cast_expr["subExpr"] = sub_expr;

  return implicit_cast_expr;
}

nlohmann::json
solidity_convertert::make_pointee_type(const nlohmann::json &sub_expr)
{
  // Since Solidity function call node does not have enough information, we need to make a JSON object
  // manually create a JSON object to complete the conversions of function to pointer decay

  // make a mapping for JSON object creation latter
  // based on the usage of get_func_decl_ref_t() in get_func_decl_ref_type()
  nlohmann::json adjusted_expr;

  if (
    sub_expr["typeString"].get<std::string>().find("function") !=
    std::string::npos)
  {
    // Add more special functions here
    if (
      sub_expr["typeString"].get<std::string>().find("function ()") !=
        std::string::npos ||
      sub_expr["typeIdentifier"].get<std::string>().find(
        "t_function_assert_pure$") != std::string::npos ||
      sub_expr["typeIdentifier"].get<std::string>().find(
        "t_function_internal_pure$") != std::string::npos)
    {
      // e.g. FunctionNoProto: "typeString": "function () returns (uint8)" with () empty after keyword 'function'
      // "function ()" contains the function args in the parentheses.
      // make a type to behave like SolidityGrammar::FunctionDeclRefT::FunctionNoProto
      // Note that when calling "assert(.)", it's like "typeIdentifier": "t_function_assert_pure$......",
      //  it's also treated as "FunctionNoProto".
      auto j2 = R"(
            {
              "nodeType": "FunctionDefinition",
              "parameters":
                {
                  "parameters" : []
                }
            }
          )"_json;
      adjusted_expr = j2;

      if (
        sub_expr["typeString"].get<std::string>().find("returns") !=
        std::string::npos)
      {
        adjusted_expr = R"(
            {
              "nodeType": "FunctionDefinition",
              "parameters":
                {
                  "parameters" : []
                }
            }
          )"_json;
        // e.g. for typeString like:
        // "typeString": "function () returns (uint8)"
        // use regex to capture the type and convert it to shorter form.
        std::smatch matches;
        std::regex e("returns \\((\\w+)\\)");
        std::string typeString = sub_expr["typeString"].get<std::string>();
        if (std::regex_search(typeString, matches, e))
        {
          auto j2 = nlohmann::json::parse(
            R"({
                "typeIdentifier": "t_)" +
            matches[1].str() + R"(",
                "typeString": ")" +
            matches[1].str() + R"("
              })");
          adjusted_expr["returnParameters"]["parameters"][0]
                       ["typeDescriptions"] = j2;
        }
        else if (
          sub_expr["typeString"].get<std::string>().find("returns (contract") !=
          std::string::npos)
        {
          // TODO: Fix me
          auto j2 = R"(
              {
                "nodeType": "ParameterList",
                "parameters": []
              }
            )"_json;
          adjusted_expr["returnParameters"] = j2;
        }
        else
          assert(!"Unsupported return types in pointee");
      }
      else
      {
        // e.g. for typeString like:
        // "typeString": "function (bool) pure"
        auto j2 = R"(
              {
                "nodeType": "ParameterList",
                "parameters": []
              }
            )"_json;
        adjusted_expr["returnParameters"] = j2;
      }
    }
    else
      assert(!"Unsupported - detected function call with parameters");
  }
  else
    assert(!"Unsupported pointee - currently we only support the semantics of function to pointer decay");

  return adjusted_expr;
}

// Parse typet object into a typeDescriptions json
nlohmann::json solidity_convertert::make_return_type_from_typet(typet type)
{
  // Useful to get the width of a int literal type for return statement
  nlohmann::json adjusted_expr;
  if (type.is_signedbv() || type.is_unsignedbv())
  {
    std::string width = type.width().as_string();
    std::string type_name = (type.is_signedbv() ? "int" : "uint") + width;
    auto j2 = nlohmann::json::parse(
      R"({
              "typeIdentifier": "t_)" +
      type_name + R"(",
              "typeString": ")" +
      type_name + R"("
            })");
    adjusted_expr = j2;
  }
  return adjusted_expr;
}

nlohmann::json solidity_convertert::make_array_elementary_type(
  const nlohmann::json &type_descrpt)
{
  // Function used to extract the type of the array and its elements
  // In order to keep the consistency and maximum the reuse of get_type_description function,
  // we used ["typeDescriptions"] instead of ["typeName"], despite the fact that the latter contains more information.
  // Although ["typeDescriptions"] also contains all the information needed, we have to do some pre-processing

  // e.g.
  //   "typeDescriptions": {
  //     "typeIdentifier": "t_array$_t_uint256_$dyn_memory_ptr",
  //     "typeString": "uint256[] memory"
  //      }
  //
  // convert to
  //
  //   "typeDescriptions": {
  //     "typeIdentifier": "t_uint256",
  //     "typeString": "uint256"
  //     }

  //! current implement does not consider Multi-Dimensional Arrays

  // 1. declare an empty json node
  nlohmann::json elementary_type;
  const std::string typeIdentifier =
    type_descrpt["typeIdentifier"].get<std::string>();

  // 2. extract type info
  // e.g.
  //  bytes[] memory x => "t_array$_t_bytes_$dyn_storage_ptr"  => t_bytes
  //  [1,2,3]          => "t_array$_t_uint8_$3_memory_ptr      => t_uint8
  assert(typeIdentifier.substr(0, 8) == "t_array$");
  std::regex rgx("\\$_\\w*_\\$");

  std::smatch match;
  if (!std::regex_search(typeIdentifier, match, rgx))
    assert(!"Cannot find array element type in typeIdentifier");
  std::string sub_match = match[0];
  std::string t_type = sub_match.substr(2, sub_match.length() - 4);
  std::string type = t_type.substr(2);

  // 3. populate node
  elementary_type = {{"typeIdentifier", t_type}, {"typeString", type}};

  return elementary_type;
}

nlohmann::json solidity_convertert::make_array_to_pointer_type(
  const nlohmann::json &type_descrpt)
{
  // Function to replace the content of ["typeIdentifier"] with "ArrayToPtr"
  // All the information in ["typeIdentifier"] should also be available in ["typeString"]
  std::string type_identifier = "ArrayToPtr";
  std::string type_string = type_descrpt["typeString"].get<std::string>();

  std::map<std::string, std::string> m = {
    {"typeIdentifier", type_identifier}, {"typeString", type_string}};
  nlohmann::json adjusted_type = m;

  return adjusted_type;
}

std::string
solidity_convertert::get_array_size(const nlohmann::json &type_descrpt)
{
  const std::string s = type_descrpt["typeString"].get<std::string>();
  std::regex rgx(".*\\[([0-9]+)\\]");
  std::string the_size;

  std::smatch match;
  if (std::regex_search(s.begin(), s.end(), match, rgx))
  {
    std::ssub_match sub_match = match[1];
    the_size = sub_match.str();
  }
  else
    assert(!"Unsupported - Missing array size in type descriptor. Detected dynamic array?");

  return the_size;
}

bool solidity_convertert::is_dyn_array(const nlohmann::json &ast_node)
{
  if (
    ast_node.contains("typeDescriptions") &&
    SolidityGrammar::get_type_name_t(ast_node["typeDescriptions"]) ==
      SolidityGrammar::DynArrayTypeName)
    return true;
  return false;
}

void solidity_convertert::get_size_of_expr(const typet &t, exprt &size_of_expr)
{
  size_of_expr = exprt("sizeof", size_type());
  typet elem_type = t;
  if (elem_type.is_struct())
  {
    struct_union_typet st = to_struct_union_type(elem_type);
    elem_type = symbol_typet(prefix + st.tag().as_string());
  }
  size_of_expr.set("#c_sizeof_type", elem_type);
}

// check if the abi.encodedSignature is the same
// note that internal/private function do not have abi signature
bool solidity_convertert::is_func_sig_cover(
  const std::string &derived,
  const std::string &base)
{
  // function signature coverage‐check lambda: name + ordered argument types
  auto covers =
    [&](const std::string &derived, const std::string &base) -> bool {
    const auto &dSigs = funcSignatures.at(derived);
    const auto &bSigs = funcSignatures.at(base);

    // Every base sig must have a matching derived sig
    for (const auto &ds : dSigs)
    {
      if (ds.name == derived)
        // skip ctor
        continue;

      if (ds.visibility == "private" || ds.visibility == "internal")
        // cannot be called via abi
        continue;

      //TODO: skip interface, abstract contract

      bool foundMatch = false;

      for (const auto &bs : bSigs)
      {
        // 1) same name?
        if (ds.name != bs.name)
          continue;

        // 2) internal or private?
        if (bs.visibility == "private" || bs.visibility == "internal")
          continue;

        // 3) same number of params?
        const auto &dArgs = to_code_type(ds.type).arguments();
        const auto &bArgs = to_code_type(bs.type).arguments();
        if (dArgs.size() != bArgs.size())
          continue;
        if (
          to_code_type(ds.type).has_ellipsis() &&
          !to_code_type(bs.type).has_ellipsis())
          continue;
        if (
          to_code_type(bs.type).has_ellipsis() &&
          !to_code_type(ds.type).has_ellipsis())
          continue;

        // 4) each parameter's type must match, in order
        bool argsMatch = true;
        for (size_t idx = 0; idx < dArgs.size(); ++idx)
        {
          if (dArgs[idx].type() != bArgs[idx].type())
          {
            argsMatch = false;
            break;
          }
        }

        if (argsMatch)
        {
          log_debug("solidity", "function {} matched", ds.name);
          foundMatch = true;
          break;
        }
      }

      // if any base‐fn had no match, derived does NOT cover base
      if (!foundMatch)
        return false;
    }

    return true;
  };

  return covers(derived, base);
}

// check if the target contract contains any public var with matched name and type, which can be accessed via abi
bool solidity_convertert::is_var_getter_matched(
  const std::string &cname,
  const std::string &tname,
  const typet &ttype)
{
  log_debug(
    "solidity",
    "heck if the target contract {} contains any public var {} with matched "
    "name and type",
    cname,
    tname);
  // 1) get contract body
  nlohmann::json contract_ref;
  for (auto &nodes : src_ast_json["nodes"])
  {
    if (
      nodes.contains("nodeType") && nodes["nodeType"] == "ContractDefinition" &&
      nodes["name"] == cname)
      contract_ref = nodes;
  }
  if (contract_ref.empty())
  {
    log_error("cannot find contract definition ref");
    abort();
  }

  const nlohmann::json body = contract_ref["nodes"];
  for (const auto &node : body)
  {
    if (
      SolidityGrammar::get_contract_body_element_t(node) ==
      SolidityGrammar::VarDecl)
    {
      assert(node.contains("visibility"));
      std::string access = node["visibility"].get<std::string>();
      if (access != "public")
        continue;

      exprt comp;
      if (get_var_decl_ref(node, false, comp))
      {
        log_error("failed to get variable reference");
        abort();
      }

      if (comp.name().as_string() == tname && comp.type() == ttype)
        return true;
    }
  }

  return false;
}

void solidity_convertert::get_unique_name(
  const std::string &name_prefix,
  const std::string &id_prefix,
  std::string &aux_name,
  std::string &aux_id)
{
  do
  {
    aux_name = name_prefix + std::to_string(aux_counter);
    aux_id = id_prefix + aux_name;
    ++aux_counter;
  } while (context.find_symbol(aux_id) != nullptr);
}

void solidity_convertert::get_aux_var(
  std::string &aux_name,
  std::string &aux_id)
{
  get_unique_name("_ESBMC_aux", "sol:@", aux_name, aux_id);
}

void solidity_convertert::get_aux_array_name(
  std::string &aux_name,
  std::string &aux_id)
{
  get_unique_name("aux_array", "sol:@", aux_name, aux_id);
}

void solidity_convertert::get_aux_array(
  const exprt &src_expr,
  const typet &sub_t,
  exprt &new_expr)
{
  log_debug("solidity", "\t\t@@@ getting auxiliary array variable");
  if (src_expr.name().as_string().find("aux_array") != std::string::npos)
  {
    // skip if it's already a aux array
    new_expr = src_expr;
    return;
  }

  // typecast for element
  exprt new_src_expr = src_expr;
  new_src_expr.type().subtype() = sub_t;
  set_sol_type(new_src_expr.type(), SolidityGrammar::SolType::ARRAY);
  for (exprt &op : new_src_expr.operands())
    solidity_gen_typecast(ns, op, sub_t);

  std::string aux_name;
  std::string aux_id;
  get_aux_array_name(aux_name, aux_id);

  locationt loc = new_src_expr.location();
  std::string debug_modulename =
    get_modulename_from_path(loc.file().as_string());

  assert(!new_src_expr.type().get("#sol_array_size").empty());
  typet t = new_src_expr.type();

  symbolt sym;
  get_default_symbol(sym, debug_modulename, t, aux_name, aux_id, loc);
  sym.static_lifetime = true;
  sym.lvalue = true;

  symbolt &added_symbol = *move_symbol_to_context(sym);

  added_symbol.value = new_src_expr;
  new_expr = symbol_expr(added_symbol);
}

void solidity_convertert::get_size_expr(const exprt &rhs, exprt &size_expr)
{
  typet rt = rhs.type();

  unsigned int arr_size = 0;
  if (!rt.get("#sol_array_size").empty())
    arr_size = std::stoi(rt.get("#sol_array_size").as_string());
  else if (rt.has_subtype() && !rt.subtype().get("#sol_array_size").empty())
    arr_size = std::stoi(rt.subtype().get("#sol_array_size").as_string());
  else
  {
    // arr_size = _ESBMC_array_length(rhs);
    side_effect_expr_function_callt length_expr;
    get_library_function_call_no_args(
      "_ESBMC_array_length",
      "c:@F@_ESBMC_array_length",
      uint_type(),
      rhs.location(),
      length_expr);
    length_expr.arguments().push_back(rhs);
    size_expr = length_expr;

    // not fall through
    return;
  }

  size_expr = constant_exprt(
    integer2binary(arr_size, bv_width(uint_type())),
    integer2string(arr_size),
    uint_type());
}

void solidity_convertert::store_update_dyn_array(
  const exprt &dyn_arr,
  const exprt &size_expr,
  exprt &store_call)
{
  // void _ESBMC_store_array(void *array, size_t length)
  side_effect_expr_function_callt length_expr;
  get_library_function_call_no_args(
    "_ESBMC_store_array",
    "c:@F@_ESBMC_store_array",
    empty_typet(),
    dyn_arr.location(),
    length_expr);
  length_expr.arguments().push_back(dyn_arr);
  length_expr.arguments().push_back(size_expr);
  store_call = length_expr;
}

// convert new array rhs
// e.g. uint* x = calloc();
bool solidity_convertert::get_empty_array_ref(
  const nlohmann::json &expr,
  exprt &new_expr)
{
  // Get Name
  nlohmann::json callee_expr_json = expr["expression"];
  nlohmann::json callee_arg_json = expr["arguments"][0];

  // Get name, id;
  std::string name, id;
  get_aux_array_name(name, id);

  // Get Location
  locationt location_begin;
  get_location_from_node(callee_expr_json, location_begin);

  // Get Type
  // 1. get elem type
  typet elem_type;
  const nlohmann::json elem_node =
    callee_expr_json["typeName"]["baseType"]["typeDescriptions"];
  if (get_type_description(elem_node, elem_type))
    return true;

  // 2. get array size
  exprt size;
  const nlohmann::json literal_type = callee_arg_json["typeDescriptions"];
  if (get_expr(callee_arg_json, literal_type, size))
    return true;

  // 3. do calloc
  side_effect_expr_function_callt calc_call;
  get_calloc_function_call(location_begin, calc_call);

  exprt size_of_expr;
  get_size_of_expr(elem_type, size_of_expr);

  calc_call.arguments().push_back(size);
  calc_call.arguments().push_back(size_of_expr);
  new_expr = calc_call;
  set_sol_type(new_expr.type(), SolidityGrammar::SolType::ARRAY_CALLOC);

  return false;
}

exprt solidity_convertert::make_aux_var(exprt &val, const locationt &location)
{
  // If val is already a symbol, no need to create an aux variable
  if (val.is_symbol())
    return val;

  std::string aux_name, aux_id;
  get_aux_var(aux_name, aux_id);

  typet t = val.type();
  std::string debug_modulename = get_modulename_from_path(absolute_path);

  symbolt aux_sym;
  get_default_symbol(aux_sym, debug_modulename, t, aux_name, aux_id, location);
  aux_sym.lvalue = true;
  aux_sym.file_local = true;

  auto &added_sym = *move_symbol_to_context(aux_sym);
  added_sym.value = val;

  code_declt decl(symbol_expr(added_sym));
  decl.operands().push_back(val);
  move_to_front_block(decl);

  return symbol_expr(added_sym);
}

// Find the last parent json node
// It will not reliably find the correct parent if the same target appears under multiple different parent nodes.
// To enusre correctness, the input is expected to contain key "id" and, if possible, "is_inherit"
const nlohmann::json &solidity_convertert::find_last_parent(
  const nlohmann::json &root,
  const nlohmann::json &target)
{
  using Frame = const nlohmann::json *; // Pointer to a node
  std::stack<Frame> stack;
  stack.push(&root);

  while (!stack.empty())
  {
    const nlohmann::json *node = stack.top();
    stack.pop();

    if (node->is_object())
    {
      for (auto it = node->begin(); it != node->end(); ++it)
      {
        const auto &value = it.value();
        if (value == target)
          return *node;

        if (value.is_structured())
          stack.push(&value);
      }
    }
    else if (node->is_array())
    {
      for (const auto &element : *node)
      {
        if (element == target)
          return *node;

        if (element.is_structured())
          stack.push(&element);
      }
    }
  }

  return empty_json;
}

// return the parent contract definition node
// return empty_json if the target_json is outside of any contract
// this function dose not rely on current_baseContractName
// so we assume that the target provided is no ambiguous
const nlohmann::json &solidity_convertert::find_parent_contract(
  const nlohmann::json &root,
  const nlohmann::json &target)
{
  using Frame = std::pair<const nlohmann::json *, const nlohmann::json *>;
  std::stack<Frame> stack;
  // Begin with the root, with no current contract context.
  stack.emplace(&root, nullptr);

  while (!stack.empty())
  {
    auto [node, current_contract] = stack.top();
    stack.pop();

    if (
      node->is_object() && node->contains("nodeType") &&
      (*node)["nodeType"] == "ContractDefinition")
    {
      current_contract = node;
    }

    // check if this node is the target.
    // We use pointer identity comparison to ensure an exact match.
    if (*node == target)
    {
      return current_contract ? *current_contract : empty_json;
    }

    // If the node is an object, iterate over its values.
    if (node->is_object())
    {
      // Use reverse order for DFS consistency.
      for (auto it = node->rbegin(); it != node->rend(); ++it)
      {
        const auto &value = it.value();
        if (value.is_structured())
          stack.emplace(&value, current_contract);
      }
    }
    // If the node is an array, do the same.
    else if (node->is_array())
    {
      for (auto it = node->rbegin(); it != node->rend(); ++it)
      {
        if (it->is_structured())
          stack.emplace(&(*it), current_contract);
      }
    }
  }

  // If the target was not found, return an empty JSON.
  return empty_json;
}

// Pure DFS: find the first node with matching "id" field in any JSON subtree.
// This is the low-level building block used by find_decl_ref and external
// callers that need unscoped lookup (e.g., during inheritance merging).
const nlohmann::json &solidity_convertert::find_node_by_id(
  const nlohmann::json &subtree,
  int ref_id)
{
  if (!subtree.is_structured())
    return empty_json;

  using Frame = const nlohmann::json *;
  std::stack<Frame> stack;
  stack.push(&subtree);

  while (!stack.empty())
  {
    const nlohmann::json *node = stack.top();
    stack.pop();

    if (node->is_object())
    {
      if (node->contains("id") && (*node)["id"] == ref_id)
        return *node;

      for (auto it = node->rbegin(); it != node->rend(); ++it)
      {
        if (it.value().is_structured())
          stack.push(&it.value());
      }
    }
    else if (node->is_array())
    {
      for (auto it = node->rbegin(); it != node->rend(); ++it)
      {
        if (it->is_structured())
          stack.push(&(*it));
      }
    }
  }

  return empty_json;
}

// Scoped declaration lookup.
// After inheritance merging, node IDs are not unique across contracts
// (inherited nodes are copied into derived contracts). This function
// restricts the search to the correct scope:
//   1. current_baseContractName (the contract being processed)
//   2. Library contracts
//   3. Global-scope nodes (structs, enums outside any contract)
// If not found, falls back to overrideMap for virtual/override resolution.
const nlohmann::json &
solidity_convertert::find_decl_ref(int ref_id)
{
  log_debug(
    "solidity",
    "\tcurrent base contract name {}, ref_id {}",
    current_baseContractName,
    std::to_string(ref_id));

  if (!src_ast_json.contains("nodes"))
    return empty_json;

  auto search_scoped = [&](int id) -> const nlohmann::json &
  {
    for (const auto &node : src_ast_json["nodes"])
    {
      if (!node.is_object())
        continue;

      bool is_contract =
        node.contains("nodeType") && node["nodeType"] == "ContractDefinition";

      if (is_contract)
      {
        // Check if the contract node itself matches
        if (node.contains("id") && node["id"] == id)
          return node;

        bool is_library =
          node.contains("contractKind") && node["contractKind"] == "library";
        bool is_base =
          !current_baseContractName.empty() && node.contains("name") &&
          node["name"] == current_baseContractName;

        // Only search inside matching contract or library
        if (is_base || is_library)
        {
          const auto &result = find_node_by_id(node, id);
          if (!result.empty())
            return result;
        }
        // Skip other contracts
      }
      else
      {
        // Global-scope node (struct, enum, etc.) — always search
        const auto &result = find_node_by_id(node, id);
        if (!result.empty())
          return result;
      }
    }
    return empty_json;
  };

  const auto &result = search_scoped(ref_id);
  if (!result.empty())
    return result;

  // Override fallback: if an inherited function was overridden,
  // redirect to the overriding function's ID
  auto override_it = overrideMap.find(current_baseContractName);
  if (override_it != overrideMap.end())
  {
    auto id_it = override_it->second.find(ref_id);
    if (id_it != override_it->second.end())
      return search_scoped(id_it->second);
  }

  return empty_json;
}
