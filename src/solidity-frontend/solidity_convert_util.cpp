/// \file solidity_convert_util.cpp
/// \brief Utility and helper functions for the Solidity converter.
///
/// Provides shared utility methods used across the converter: source location
/// extraction, AST node search by ID, parent node lookup, contract name
/// resolution, line number computation from source ranges, JSON AST traversal
/// helpers, and various name/ID construction routines.

#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <functional>
#include <util/arith/arith_tools.h>
#include <util/arith/bitvector.h>
#include <util/lang/c_sizeof.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <util/base/i2string.h>
#include <util/arith/mp_arith.h>
#include <util/irep/std_expr.h>
#include <util/message/message.h>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <regex>
#include <optional>

#include <fstream>

// Lexically-declaring contract of an AST node: the top-level
// ContractDefinition whose source byte span contains the node's `src`
// start offset. This is the definition of "lexical declarer" itself and
// is invariant across inheritance merge-by-copy and modifier splicing
// (merge/splice copy the json verbatim including `src`, so a decision
// keeps its original textual offset = inside its declaring contract's
// span). Uniform for FunctionDefinition AND ModifierDefinition (the
// latter has no `scope` field). Done on the AST at stamp time — not on
// goto `it->location` line ranges (that was the unsound rejected Fix-A,
// a different layer). Empty for file-scope/free constructs (correctly
// unattributed → excluded under --contract). Contract spans do not
// overlap (siblings in the flattened SourceUnit).
const std::string &
solidity_convertert::current_decl_contract(const nlohmann::json &ast_node)
{
  static const std::string empty;
  if (!cd_id_to_name_built)
  {
    cd_id_to_name_built = true;
    if (src_ast_json.contains("nodes") && src_ast_json["nodes"].is_array())
      for (const auto &n : src_ast_json["nodes"])
        if (
          n.is_object() && n.value("nodeType", "") == "ContractDefinition" &&
          n.contains("name") && n.contains("src"))
        {
          const std::string s = n["src"].get<std::string>();
          size_t c1 = s.find(':');
          size_t c2 = s.find(':', c1 + 1);
          long st = std::stol(s.substr(0, c1));
          long ln = std::stol(s.substr(c1 + 1, c2 - c1 - 1));
          cd_spans.push_back({st, st + ln, n["name"].get<std::string>()});
        }
  }

  if (!ast_node.is_object())
    return empty;
  const std::string src = ast_node.contains("src")
                            ? ast_node["src"].get<std::string>()
                            : get_src_from_json(ast_node);
  if (src.empty() || src.find(':') == std::string::npos)
    return empty;
  long off = std::stol(src.substr(0, src.find(':')));
  for (const auto &sp : cd_spans)
    if (off >= sp.start && off < sp.end)
      return sp.name;
  return empty;
}

void solidity_convertert::get_location_from_node(
  const nlohmann::json &ast_node,
  locationt &location)
{
  location.set_line(get_line_number(ast_node));
  location.set_file(
    absolute_path); // assume absolute_path is the name of the contrace file,
                    // since we ran solc in the same directory

  // To annotate local declaration within a function
  if (current_functionDecl)
  {
    location.set_function(current_functionName); // set the function where this
                                                 // local variable belongs to
    const std::string &dc = current_decl_contract(ast_node);
    if (!dc.empty())
      location.set("sol_decl_contract", dc);
  }
}

void solidity_convertert::get_start_location_from_stmt(
  const nlohmann::json &ast_node,
  locationt &location)
{
  std::string function_name;

  if (current_functionDecl)
    function_name = current_functionName;

  // The line number is derived from the Solidity AST JSON source range.
  location.set_line(get_line_number(ast_node));
  location.set_file(
    absolute_path); // assume absolute_path is the name of the contrace file,
                    // since we ran solc in the same directory

  if (!function_name.empty())
    location.set_function(function_name);

  if (current_functionDecl)
  {
    const std::string &dc = current_decl_contract(ast_node);
    if (!dc.empty())
      location.set("sol_decl_contract", dc);
  }
}

void solidity_convertert::get_final_location_from_stmt(
  const nlohmann::json &ast_node,
  locationt &location)
{
  std::string function_name;

  if (current_functionDecl)
    function_name = current_functionName;

  // The line number is derived from the Solidity AST JSON source range.
  location.set_line(get_line_number(ast_node, true));
  location.set_file(
    absolute_path); // assume absolute_path is the name of the contrace file,
                    // since we ran solc in the same directory

  if (!function_name.empty())
    location.set_function(function_name);

  if (current_functionDecl)
  {
    const std::string &dc = current_decl_contract(ast_node);
    if (!dc.empty())
      location.set("sol_decl_contract", dc);
  }
}

unsigned int solidity_convertert::get_line_number(
  const nlohmann::json &ast_node,
  bool final_position)
{
  // Solidity src means "start:length:index", where "start" represents the
  // position of the first char byte of the identifier.
  std::string src = ast_node.contains("src")
                      ? ast_node["src"].get<std::string>()
                      : get_src_from_json(ast_node);

  std::string position = src.substr(0, src.find(":"));
  // size_t, not unsigned int: narrowing here would wrap an out-of-range offset
  // back into the buffer and silently bypass the clamp below.
  size_t byte_position = std::stoul(position) + 1;

  if (final_position)
    byte_position = add_offset(src, byte_position);

  // A `src` range routinely points past the end of contract_contents -- most
  // often because add_offset below mis-reads the range length, but also when
  // the .solast was generated from another revision of the source or from a
  // multi-source compilation. Walking there is an out-of-bounds read whose
  // result varies per run (esbmc/esbmc#6760); clamping reports the line just
  // past the last one instead.
  byte_position = std::min(byte_position, contract_contents.size());

  // the line number can be calculated by counting the number of line breaks
  // prior to the identifier.
  // Clamp byte_position to contract_contents.size() to avoid
  // heap-buffer-overflow when AST nodes carry synthetic/out-of-range src
  // offsets (observed on auxiliary vars populated before typechecking in large
  // import closures).
  if (byte_position > contract_contents.size())
    byte_position = contract_contents.size();
  unsigned int loc = std::count(
                       contract_contents.begin(),
                       (contract_contents.begin() + byte_position),
                       '\n') +
                     1;
  return loc;
}

size_t
solidity_convertert::add_offset(const std::string &src, size_t start_position)
{
  // extract the length from "start:length:index"
  // The previous implementation was `src.substr(1, src.find(":"))`,
  // which only worked by accident for multi-digit start positions and
  // crashed (`stoul` invalid_argument) when the start digit count made
  // the substring begin at or past the first colon (e.g. src="5:3:0"
  // yielded ":" → stoul throws). Properly slice the *second* field.
  const auto first = src.find(":");
  if (first == std::string::npos)
    return start_position;
  const auto second = src.find(":", first + 1);
  const std::string offset = (second == std::string::npos)
                               ? src.substr(first + 1)
                               : src.substr(first + 1, second - first - 1);
  if (offset.empty() || !std::isdigit(static_cast<unsigned char>(offset[0])))
    return start_position;
  return start_position + std::stoul(offset);
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

  return path; // for _x, it just returns "overflow_2.c" because the test
               // program is in the same dir as esbmc binary
}

bool solidity_convertert::get_constant_value(
  const int ref_id,
  std::string &value)
{
  log_debug("solidity", "get constant var's value");
  nlohmann::json tmp = find_node_by_id(src_ast_json, ref_id);
  while (!tmp.empty() && tmp.contains("value"))
  {
    const auto &val_json = tmp["value"];
    if (
      val_json.is_object() && val_json.contains("value") &&
      val_json["value"].is_string())
    {
      value = val_json["value"].get<std::string>();
      return false;
    }
    // Follow simple Identifier chains; anything else (TupleExpression,
    // BinaryOperation, ...) is not resolvable here.
    if (
      val_json.is_object() && val_json.contains("referencedDeclaration") &&
      val_json["referencedDeclaration"].is_number_integer())
    {
      int new_ref_id = val_json["referencedDeclaration"].get<int>();
      if (new_ref_id <= 0)
        return true;
      tmp = find_node_by_id(src_ast_json, new_ref_id);
      continue;
    }
    return true;
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
  symbol.set_type(std::move(type));
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
  // function to detect special intrinsic functions, e.g. ___ESBMC_assume.
  // __ESBMC_reverted is a verification-only revert-observation stub: the user
  // declares an empty body so solc can compile, and the frontend hijacks calls
  // to the C intrinsic (get_sol_builtin_ref).  Treat it as intrinsic here so
  // the dead empty body is not materialized.
  return (
    ast_node.contains("name") && (ast_node["name"] == "__ESBMC_assume" ||
                                  ast_node["name"] == "__VERIFIER_assume" ||
                                  ast_node["name"] == "__ESBMC_assert" ||
                                  ast_node["name"] == "__VERIFIER_assert" ||
                                  ast_node["name"] == "__ESBMC_reverted"));
}

nlohmann::json solidity_convertert::make_implicit_cast_expr(
  const nlohmann::json &sub_expr,
  std::string cast_type)
{
  log_debug("solidity", "\t@@@ make_implicit_cast_expr");
  // Since Solidity AST does not have type cast information about return values,
  // we need to manually make a JSON object and wrap the return expression in
  // it.
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
  // Since Solidity function call node does not have enough information, we need
  // to make a JSON object manually create a JSON object to complete the
  // conversions of function to pointer decay

  // make a mapping for JSON object creation latter
  // based on the usage of get_func_decl_ref_t() in get_func_decl_ref_type()
  nlohmann::json adjusted_expr;

  if (
    sub_expr["typeString"].get<std::string>().find("function") !=
    std::string::npos)
  {
    // Match all function types by their typeIdentifier prefix.
    // Function types with parameters (e.g. function(uint) pure returns(uint))
    // are handled as FunctionNoProto (losing param info) — sufficient for
    // FunctionToPointer decay even though indirect calls are not yet supported.
    if (
      sub_expr["typeString"].get<std::string>().find("function ()") !=
        std::string::npos ||
      sub_expr["typeIdentifier"].get<std::string>().find("t_function_") !=
        std::string::npos)
    {
      // e.g. FunctionNoProto: "typeString": "function () returns (uint8)" with
      // () empty after keyword 'function' "function ()" contains the function
      // args in the parentheses. make a type to behave like
      // SolidityGrammar::FunctionDeclRefT::FunctionNoProto Note that when
      // calling "assert(.)", it's like "typeIdentifier":
      // "t_function_assert_pure$......",
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
          auto rtn_params = R"(
              {
                "nodeType": "ParameterList",
                "parameters": [
                  {
                    "typeDescriptions": {}
                  }
                ]
              }
            )"_json;
          rtn_params["parameters"][0]["typeDescriptions"] = j2;
          adjusted_expr["returnParameters"] = rtn_params;
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
  // In order to keep the consistency and maximum the reuse of
  // get_type_description function, we used ["typeDescriptions"] instead of
  // ["typeName"], despite the fact that the latter contains more information.
  // Although ["typeDescriptions"] also contains all the information needed, we
  // have to do some pre-processing

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
  const std::string typeString =
    type_descrpt.contains("typeString")
      ? type_descrpt["typeString"].get<std::string>()
      : "";

  // 2. extract element identifier by scanning the typeIdentifier from the
  // back for the outer array's size delimiter "_$<dyn|digits>".  Anything
  // before it is the element type identifier (which may itself carry a
  // location suffix for reference types like string/bytes/struct).
  // Examples:
  //   t_array$_t_uint256_$dyn_memory_ptr           → t_uint256
  //   t_array$_t_string_memory_ptr_$2_memory_ptr   → t_string_memory_ptr
  //   t_array$_t_bytes_$dyn_storage_ptr            → t_bytes
  // Guard: callers occasionally hand us a typeDescriptions whose
  // identifier is not an array (e.g. degenerate tuple contexts such as
  // `return (a, [1,2,3][0])`, where the element typeDescriptions bubbled
  // up one level). Return an empty element descriptor so the caller's
  // recovery path (typeString/fallback) kicks in instead of crashing
  // inside substr below.
  if (typeIdentifier.compare(0, 9, "t_array$_") != 0)
  {
    elementary_type = {
      {"typeIdentifier", typeIdentifier}, {"typeString", typeString}};
    return elementary_type;
  }
  const std::string prefix = "t_array$_";
  std::string rest = typeIdentifier.substr(prefix.size());

  std::string elem_id;
  size_t last = rest.rfind("_$");
  while (last != std::string::npos)
  {
    std::string after = rest.substr(last + 2);
    if (
      after.compare(0, 3, "dyn") == 0 ||
      (!after.empty() && std::isdigit(static_cast<unsigned char>(after[0]))))
    {
      elem_id = rest.substr(0, last);
      break;
    }
    if (last < 2)
      break;
    last = rest.rfind("_$", last - 1);
  }
  if (elem_id.empty())
    elem_id = rest;

  // 3. strip trailing data-location suffix from the element identifier so it
  // maps back to a plain elementary/reference type recognised by
  // get_type_name_t (e.g. t_string_memory_ptr → t_string).
  static const char *const loc_suffixes[] = {
    "_memory_ptr",
    "_storage_ptr",
    "_calldata_ptr",
    "_memory",
    "_storage",
    "_calldata"};
  for (const char *suf : loc_suffixes)
  {
    size_t slen = std::strlen(suf);
    if (
      elem_id.size() > slen &&
      elem_id.compare(elem_id.size() - slen, slen, suf) == 0)
    {
      elem_id.erase(elem_id.size() - slen);
      break;
    }
  }

  // 4. derive element typeString: strip data-location qualifier and the
  // trailing "[<size>]" from the array typeString.
  std::string elem_ts = typeString;
  auto strip_loc = [](std::string &s) {
    for (const char *suf :
         {" storage ref",
          " storage pointer",
          " storage",
          " memory",
          " calldata"})
    {
      size_t slen = std::strlen(suf);
      if (s.size() > slen && s.compare(s.size() - slen, slen, suf) == 0)
      {
        s.erase(s.size() - slen);
        return;
      }
    }
  };
  strip_loc(elem_ts);
  auto last_bracket = elem_ts.rfind('[');
  if (last_bracket != std::string::npos)
    elem_ts.erase(last_bracket);
  strip_loc(elem_ts);
  if (elem_ts.empty() && elem_id.compare(0, 2, "t_") == 0)
    elem_ts = elem_id.substr(2);

  // 5. populate node
  elementary_type = {{"typeIdentifier", elem_id}, {"typeString", elem_ts}};

  return elementary_type;
}

nlohmann::json solidity_convertert::make_array_to_pointer_type(
  const nlohmann::json &type_descrpt)
{
  // Function to replace the content of ["typeIdentifier"] with "ArrayToPtr"
  // All the information in ["typeIdentifier"] should also be available in
  // ["typeString"]
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
  if (!ast_node.contains("typeDescriptions"))
    return false;
  auto type = SolidityGrammar::get_type_name_t(ast_node["typeDescriptions"]);
  if (type == SolidityGrammar::DynArrayTypeName)
    return true;
  if (
    type == SolidityGrammar::NestedArrayTypeName &&
    ast_node.contains("nodeType") && ast_node["nodeType"] == "ArrayTypeName" &&
    !ast_node.contains("length"))
    return true;
  return false;
}

void solidity_convertert::get_size_of_expr(const typet &t, exprt &size_of_expr)
{
  // Emit a `sizeof` node carrying the measured type as the type of its first
  // operand (a type_exprt) and the byte size as a second operand, mirroring the
  // C frontend; the type lets get_alloc_type recover the allocated type and the
  // value gives a sound, fully-evaluated size (esbmc/esbmc#5337).
  typet elem_type = t;
  if (elem_type.is_struct())
  {
    struct_union_typet st = to_struct_union_type(elem_type);
    elem_type = symbol_typet(prefix + st.tag().as_string());
  }
  exprt value = c_sizeof(elem_type, ns);
  if (value.is_nil())
  {
    log_error("sizeof: type has no size, {}", elem_type.name());
    abort();
  }
  size_of_expr = exprt("sizeof", size_type());
  size_of_expr.copy_to_operands(type_exprt(elem_type));
  size_of_expr.copy_to_operands(value);
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

      // TODO: skip interface, abstract contract

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

// check if the target contract contains any public var with matched name and
// type, which can be accessed via abi
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

  assert(has_sol_array_size(new_src_expr.type()));
  typet t = new_src_expr.type();

  symbolt sym;
  get_default_symbol(sym, debug_modulename, t, aux_name, aux_id, loc);
  sym.static_lifetime = true;
  sym.lvalue = true;

  symbolt &added_symbol = *move_symbol_to_context(sym);

  added_symbol.set_value(new_src_expr);
  new_expr = symbol_expr(added_symbol);
}

void solidity_convertert::get_size_expr(const exprt &rhs, exprt &size_expr)
{
  typet rt = rhs.type();

  unsigned int arr_size = 0;
  if (has_sol_array_size(rt))
    arr_size = std::stoi(get_sol_array_size(rt));
  else if (rt.has_subtype() && has_sol_array_size(rt.subtype()))
    arr_size = std::stoi(get_sol_array_size(rt.subtype()));
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

// T1.1 Stage S2: emit `_ESBMC_dynarr_idx(this->$address, pos)` so element
// reads/writes on state-var dyn-arrays are addr-keyed.  See declaration
// in solidity_convert.h.
bool solidity_convertert::get_dynarr_elem_idx(const exprt &pos, exprt &out)
{
  exprt this_expr;
  if (current_functionDecl)
  {
    if (get_func_decl_this_ref(*current_functionDecl, this_expr))
      return true;
  }
  else if (!current_baseContractName.empty())
  {
    if (get_ctor_decl_this_ref(current_baseContractName, this_expr))
      return true;
  }
  else
  {
    log_error("get_dynarr_elem_idx: no current function or contract context");
    return true;
  }

  exprt addr_expr = member_exprt(this_expr, "$address", addr_t);

  side_effect_expr_function_callt fold_call;
  get_library_function_call_no_args(
    "_ESBMC_dynarr_idx",
    "c:@F@_ESBMC_dynarr_idx",
    unsignedbv_typet(64),
    pos.location(),
    fold_call);
  fold_call.arguments().push_back(addr_expr);

  exprt pos_u256 = pos;
  solidity_gen_typecast(ns, pos_u256, unsignedbv_typet(256));
  fold_call.arguments().push_back(pos_u256);

  out = fold_call;
  return false;
}

// T1.1 Stage S1: addr-keyed dyn-array length helper.  See
// solidity_convert.h:get_dynarr_len_ref docstring.
bool solidity_convertert::get_dynarr_len_ref(const symbolt &len_sym, exprt &out)
{
  exprt this_expr;
  if (current_functionDecl)
  {
    if (get_func_decl_this_ref(*current_functionDecl, this_expr))
      return true;
  }
  else if (!current_baseContractName.empty())
  {
    if (get_ctor_decl_this_ref(current_baseContractName, this_expr))
      return true;
  }
  else
  {
    log_error(
      "get_dynarr_len_ref: no current function or contract context for {}",
      len_sym.name.as_string());
    return true;
  }

  exprt addr_expr = member_exprt(this_expr, "$address", addr_t);
  out = index_exprt(symbol_expr(len_sym), addr_expr, unsignedbv_typet(256));
  return false;
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

// Detect `new C(...)` initialization for any variable declaration id,
// covering both shapes:
//   1. State var:  VariableDeclaration { id, value: FunctionCall(NewExpr) }
//   2. Local var:  VariableDeclarationStatement {
//                    declarations: [{ id }],
//                    initialValue: FunctionCall(NewExpr) }
// Walks src_ast_json once per call.  Used by the auto-bind logic so cross-
// contract calls on `new`-created locals execute the callee body in unbound
// mode (matching SMTChecker).
bool solidity_convertert::is_new_created_decl(int decl_id) const
{
  // Recognise both shapes:
  //   new C(args)            -> FunctionCall(NewExpression, args)
  //   new C{value: v}(args)  ->
  //   FunctionCall(FunctionCallOptions(NewExpression), args)
  auto is_new_call = [](const nlohmann::json &v) {
    if (
      !v.is_object() || v.value("nodeType", "") != "FunctionCall" ||
      !v.contains("expression"))
      return false;
    const auto &inner = v["expression"];
    if (!inner.is_object())
      return false;
    if (inner.value("nodeType", "") == "NewExpression")
      return true;
    if (
      inner.value("nodeType", "") == "FunctionCallOptions" &&
      inner.contains("expression") && inner["expression"].is_object() &&
      inner["expression"].value("nodeType", "") == "NewExpression")
      return true;
    return false;
  };

  std::function<bool(const nlohmann::json &)> walk =
    [&](const nlohmann::json &node) -> bool {
    if (node.is_object())
    {
      const std::string nt = node.value("nodeType", "");
      if (nt == "VariableDeclaration" && node.value("id", -1) == decl_id)
      {
        if (node.contains("value") && is_new_call(node["value"]))
          return true;
      }
      if (nt == "VariableDeclarationStatement" && node.contains("declarations"))
      {
        for (const auto &d : node["declarations"])
        {
          if (d.is_object() && d.value("id", -1) == decl_id)
          {
            if (
              node.contains("initialValue") &&
              is_new_call(node["initialValue"]))
              return true;
          }
        }
      }
      // Assignment form: `p = new C(...)` in a function body (e.g. a state var
      // declared `C p;` then assigned in the constructor — the shape the R2
      // differential harness uses).  The decl carries no `value`, so the two
      // branches above miss it; recognise the Assignment whose LHS Identifier
      // references decl_id and whose RHS is a `new` call.  Without this, a
      // getter on such an instance is treated as a truly-unbound external call
      // and havoc'd to nondet under --unbound (false differences).
      if (
        nt == "Assignment" && node.value("operator", "") == "=" &&
        node.contains("leftHandSide") && node.contains("rightHandSide"))
      {
        const auto &lhs = node["leftHandSide"];
        if (
          lhs.is_object() && lhs.value("nodeType", "") == "Identifier" &&
          lhs.value("referencedDeclaration", -1) == decl_id &&
          is_new_call(node["rightHandSide"]))
          return true;
      }
      for (auto it = node.begin(); it != node.end(); ++it)
        if (walk(it.value()))
          return true;
    }
    else if (node.is_array())
    {
      for (const auto &c : node)
        if (walk(c))
          return true;
    }
    return false;
  };
  return walk(src_ast_json);
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

  // 3. do alloc — pick calloc (zero-init) for constant sizes, malloc
  // (symbolic-safe) otherwise. calloc's internal memset loop creates a
  // VLA-typed backing when the size is symbolic, which loses indexed
  // write/read tracking through later pointer reassignment.
  side_effect_expr_function_callt calc_call;
  if (size.is_constant())
    get_calloc_function_call(location_begin, calc_call);
  else
    get_malloc_array_function_call(location_begin, calc_call);

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
  // The value-side type can carry a `#sol_state_var = "1"` tag inherited
  // from a state-var operand (e.g. `bytes(s)` where `s` is a state var).
  // Aux vars are file-local by construction, so strip the tag — otherwise
  // the constructor-init walker (solidity_convert_constructor.cpp) treats
  // the aux as a state-var member and emits `this->_ESBMC_auxN`, which
  // doesn't exist on the contract struct and aborts goto generation.
  if (!t.get("#sol_state_var").empty())
    t.remove("#sol_state_var");
  std::string debug_modulename = get_modulename_from_path(absolute_path);

  symbolt aux_sym;
  get_default_symbol(aux_sym, debug_modulename, t, aux_name, aux_id, location);
  aux_sym.lvalue = true;
  aux_sym.file_local = true;

  auto &added_sym = *move_symbol_to_context(aux_sym);
  added_sym.set_value(migrate_expr(val));

  code_declt decl(symbol_expr(added_sym));
  decl.operands().push_back(val);
  move_to_front_block(decl);

  return symbol_expr(added_sym);
}

// Find the last parent json node
// It will not reliably find the correct parent if the same target appears under
// multiple different parent nodes. To enusre correctness, the input is expected
// to contain key "id" and, if possible, "is_inherit"
const nlohmann::json &solidity_convertert::find_last_parent(
  const nlohmann::json &root,
  const nlohmann::json &target)
{
  // Walks the AST tree to find the closest parent OBJECT that contains
  // `target` (as a direct field value, or as an element of one of its
  // array fields). Always returns an object, never an array — callers
  // expect to look up keys like `arguments`, `src`, etc. on the result.
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
        if (value.is_array())
        {
          for (const auto &element : value)
            if (element == target)
              return *node;
        }
        if (value.is_structured())
          stack.push(&value);
      }
    }
    else if (node->is_array())
    {
      // Arrays should normally be reached via their owning object so the
      // object-branch above already inspected them. Keep this branch only
      // to recurse into nested structured elements when the root itself
      // happens to be an array.
      for (const auto &element : *node)
      {
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
  // Lazy memo keyed by the node's exact serialised content. This
  // function's result is a pure function of `target` CONTENT (the DFS
  // returns the enclosing contract of the FIRST node that deep-`==`s
  // target; `root` is invariant — every caller passes
  // src_ast_json["nodes"]). The key MUST be the full content: the
  // Solidity frontend mutates AST nodes in place during conversion
  // (implicit-cast insertion, type annotation), so a node's (src,id)
  // is NOT a stable content identity across calls — only the
  // serialised value captures content-at-call-time. Two nodes with
  // equal content have equal dumps and the DFS yields the same first
  // match for both, so caching by dump is bug-for-bug identical to the
  // uncached DFS, and robust to copies / sub-references / synthetic
  // nodes. Cached ContractDefinition pointers stay valid for the run
  // (top-level contract objects never relocate). Cleared per
  // convert() run.
  const std::string key = target.dump();
  auto memo_it = fpc_memo.find(key);
  if (memo_it != fpc_memo.end())
    return memo_it->second ? *(memo_it->second) : empty_json;

  const nlohmann::json *result = nullptr; // enclosing contract, or none

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

    // Match by DEEP VALUE EQUALITY, not pointer identity. (A prior
    // comment here wrongly claimed "pointer identity"; the code always
    // did `*node == target`.) This is load-bearing, not an oversight:
    // get_current_contract_name() and inheritance resolution depend on
    // returning the enclosing contract of the FIRST content-equal node
    // in DFS order — callers routinely pass content-copies / merged /
    // synthetic nodes whose address differs from the AST original, and
    // switching to `node == &target` regresses inherited/merged-member
    // cases. Do not "fix" this to pointer identity.
    if (*node == target)
    {
      result = current_contract;
      break;
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

  fpc_memo.emplace(key, result);
  return result ? *result : empty_json;
}

// Pure DFS: find the first node with matching "id" field in any JSON subtree.
// This is the low-level building block used by find_decl_ref and external
// callers that need unscoped lookup (e.g., during inheritance merging).
const nlohmann::json &
solidity_convertert::find_node_by_id(const nlohmann::json &subtree, int ref_id)
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
const nlohmann::json &solidity_convertert::find_decl_ref(int ref_id)
{
  log_debug(
    "solidity",
    "\tcurrent base contract name {}, ref_id {}",
    current_baseContractName,
    std::to_string(ref_id));

  if (!src_ast_json.contains("nodes"))
    return empty_json;

  auto search_scoped = [&](int id) -> const nlohmann::json & {
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
        bool is_interface =
          node.contains("contractKind") && node["contractKind"] == "interface";
        bool is_base = !current_baseContractName.empty() &&
                       node.contains("name") &&
                       node["name"] == current_baseContractName;

        // Search inside the current base, libraries, and interfaces.
        // Interfaces routinely host struct definitions that are referenced
        // by member access from other contracts (e.g. 1inch limit-order-
        // protocol: `struct Order` lives in `interface IOrderMixin` and
        // every extension accesses `order.maker` through it).
        if (is_base || is_library || is_interface)
        {
          const auto &result = find_node_by_id(node, id);
          if (!result.empty())
            return result;
        }
        // Skip other non-base, non-interface, non-library contracts
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
