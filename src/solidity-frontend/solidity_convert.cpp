#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/typecast.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <iomanip>

solidity_convertert::solidity_convertert(contextt &_context,
    nlohmann::json &_ast_json, const std::string &_sol_func, const messaget &msg):
    context(_context),
    ns(context),
    ast_json(_ast_json),
    sol_func(_sol_func),
    msg(msg)
    //current_functionDecl(nullptr)
{
}

bool solidity_convertert::convert()
{
  // This function consists of two parts:
  //  1. First, we perform pattern-based verificaiton
  //  2. Then we populate the context with symbols annotated based on the each AST node, and hence prepare for the GOTO conversion.

  if (!ast_json.contains("nodes")) // check json file contains AST nodes as Solidity might change
    assert(!"JSON file does not contain any AST nodes");

  if (!ast_json.contains("absolutePath")) // check json file contains AST nodes as Solidity might change
    assert(!"JSON file does not contain absolutePath");

  absolute_path = ast_json["absolutePath"].get<std::string>();

  // By now the context should have the symbols of all ESBMC's intrinsics and the dummy main
  // We need to convert Solidity AST nodes to the equivalent symbols and add them to the context
  nlohmann::json nodes = ast_json["nodes"];

  bool found_contract_def = false;
  unsigned index = 0;
  nlohmann::json::iterator itr = nodes.begin();
  for (; itr != nodes.end(); ++itr, ++index)
  {
    // ignore the meta information and locate nodes in ContractDefinition
    std::string node_type = (*itr)["nodeType"].get<std::string>();
    if (node_type == "ContractDefinition") // contains AST nodes we need
    {
      found_contract_def = true;
      // pattern-based verification
      assert(itr->contains("nodes"));
      auto pattern_check = std::make_unique<pattern_checker>((*itr)["nodes"], sol_func, msg);
      if(pattern_check->do_pattern_check())
        return true; // 'true' indicates something goes wrong.
    }
  }
  assert(found_contract_def);

  // reasoning-based verification
  //assert(!"Continue with symbol annotations");
  index = 0;
  itr = nodes.begin();
  for (; itr != nodes.end(); ++itr, ++index)
  {
    std::string node_type = (*itr)["nodeType"].get<std::string>();
    if (node_type == "ContractDefinition") // rule source-unit
    {
      if(convert_ast_nodes(*itr))
        return true; // 'true' indicates something goes wrong.
    }
  }

  return false; // 'false' indicates successful completion.
}

bool solidity_convertert::convert_ast_nodes(const nlohmann::json &contract_def)
{
  unsigned index = 0;
  nlohmann::json ast_nodes = contract_def["nodes"];
  nlohmann::json::iterator itr = ast_nodes.begin();
  for (; itr != ast_nodes.end(); ++itr, ++index)
  {
    nlohmann::json ast_node = *itr;
    std::string node_name = ast_node["name"].get<std::string>();
    std::string node_type = ast_node["nodeType"].get<std::string>();
    printf("@@ Converting node[%u]: name=%s, nodeType=%s ...\n",
        index, node_name.c_str(), node_type.c_str());
    print_json_array_element(ast_node, node_type, index);
    exprt dummy_decl;
    if(get_decl(ast_node, dummy_decl))
      return true;
  }

  return false;
}

bool solidity_convertert::get_decl(const nlohmann::json &ast_node, exprt &new_expr)
{
  new_expr = code_skipt();

  if (!ast_node.contains("nodeType"))
    assert(!"Missing \'nodeType\' filed in ast_node");

  SolidityGrammar::ContractBodyElementT type = SolidityGrammar::get_contract_body_element_t(ast_node);

  // based on each element as in Solidty grammar "rule contract-body-element"
  switch(type)
  {
    case SolidityGrammar::ContractBodyElementT::StateVarDecl:
    {
      return get_state_var_decl(ast_node, new_expr); // rule state-variable-declaration
    }
    default:
    {
      assert(!"Unimplemented type in rule contract-body-element");
      return true;
    }
  }

  assert(!"all AST node done?");

  return false;
}

bool solidity_convertert::get_state_var_decl(const nlohmann::json &ast_node, exprt &new_expr)
{
  // For Solidity rule state-variable-declaration:
  // 1. populate typet
  typet t;
  if (get_type_name(ast_node["typeName"], t)) // "type-name" as in state-variable-declaration
    return true;

  // 2. populate id and name
  std::string name, id;
  get_state_var_decl_name(ast_node, name, id);

  // 3. populate location
  locationt location_begin;
  get_location_from_decl(ast_node, location_begin);

  // 4. populate debug module name
  std::string debug_modulename = get_modulename_from_path(absolute_path);

  // 5. set symbol attributes
  symbolt symbol;
  get_default_symbol(
    symbol,
    debug_modulename,
    t,
    name,
    id,
    location_begin);

  symbol.lvalue = true;
  symbol.static_lifetime = true; // hard coded for now, may need to change later
  symbol.is_extern = false;
  symbol.file_local = false;

  bool has_init = ast_node.contains("value");
  if(symbol.static_lifetime && !symbol.is_extern && !has_init)
  {
    symbol.value = gen_zero(t, true);
    symbol.value.zero_initializer(true);
  }

  // 6. add symbol into the context
  // just like clang-c-frontend, we have to add the symbol before converting the initial assignment
  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // 7. populate init value if there is any
  code_declt decl(symbol_expr(added_symbol));

  if (has_init)
  {
    assert(!"unimplemented - state var decl has init value");
  }

  decl.location() = location_begin;

  new_expr = decl;

  return false;
}

bool solidity_convertert::get_type_name(const nlohmann::json &type_name, typet &new_type)
{
  // For Solidity rule type-name:
  SolidityGrammar::TypeNameT type = SolidityGrammar::get_type_name_t(type_name);

  switch(type)
  {
    case SolidityGrammar::TypeNameT::ElementaryTypeName:
    {
      return get_elementary_type_name(type_name, new_type); // rule state-variable-declaration
    }
    default:
    {
      assert(!"Unimplemented type in rule type-name");
      return true;
    }
  }

  // TODO: More var decl attributes checks:
  //    - Constant
  //    - Volatile
  //    - isRestrict

  return false;
}

bool solidity_convertert::get_elementary_type_name(const nlohmann::json &type_name, typet &new_type)
{
  // For Solidity rule elementary-type-name:
  // equivalent to clang's get_builtin_type()
  std::string c_type;
  SolidityGrammar::ElementaryTypeNameT type = SolidityGrammar::get_elementary_type_name_t(type_name);

  switch(type)
  {
    // rule unsigned-integer-type
    case SolidityGrammar::ElementaryTypeNameT::UINT8:
    {
      new_type = unsigned_char_type();
      c_type = "unsigned_char";
      new_type.set("#cpp_type", c_type);
      break;
    }
    default:
    {
      assert(!"Unimplemented type in rule elementary-type-name");
      return true;
    }
  }

  return false;
}

void solidity_convertert::get_state_var_decl_name(
    const nlohmann::json &ast_node,
    std::string &name, std::string &id)
{
  // Follow the way in clang:
  //  - For state variable name, just use the ast_node["name"]
  //  - For state variable id, add prefix "c:@"
  name = ast_node["name"].get<std::string>(); // assume Solidity AST json object has "name" field, otherwise throws an exception in nlohmann::json
  id = "c:@" + name;
}

void solidity_convertert::get_location_from_decl(const nlohmann::json &ast_node, locationt &location)
{
  // The src manager of Solidity AST JSON is too encryptic.
  // For the time being we are setting it to "1".
  location.set_line(1);
  location.set_file(absolute_path); // assume absolute_path is the name of the contrace file, since we ran solc in the same directory
}

std::string solidity_convertert::get_modulename_from_path(std::string path)
{
  std::string filename = get_filename_from_path(path);

  if(filename.find_last_of('.') != std::string::npos)
    return filename.substr(0, filename.find_last_of('.'));

  return filename;
}

std::string solidity_convertert::get_filename_from_path(std::string path)
{
  if(path.find_last_of('/') != std::string::npos)
    return path.substr(path.find_last_of('/') + 1);

  return path; // for _x, it just returns "overflow_2.c" because the test program is in the same dir as esbmc binary
}

void solidity_convertert::get_default_symbol(
  symbolt &symbol,
  std::string module_name,
  typet type,
  std::string name,
  std::string id,
  locationt location)
{
  symbol.mode = "C";
  symbol.module = module_name;
  symbol.location = std::move(location);
  symbol.type = std::move(type);
  symbol.name = name;
  symbol.id = id;
}

symbolt *solidity_convertert::move_symbol_to_context(symbolt &symbol)
{
  symbolt *s = context.find_symbol(symbol.id);
  if(s == nullptr)
  {
    if(context.move(symbol, s))
    {
      std::cerr << "Couldn't add symbol " << symbol.name
                << " to symbol table\n";
      symbol.dump();
      abort();
    }
  }
  else
  {
    // types that are code means functions
    if(s->type.is_code())
    {
      if(symbol.value.is_not_nil() && !s->value.is_not_nil())
        s->swap(symbol);
    }
    else if(s->is_type)
    {
      if(symbol.type.is_not_nil() && !s->type.is_not_nil())
        s->swap(symbol);
    }
  }

  return s;
}

void solidity_convertert::print_json_element(nlohmann::json &json_in, const unsigned index,
    const std::string &key, const std::string& json_name)
{
  printf("### %s element[%u] content: key=%s, size=%lu ###\n",
      json_name.c_str(), index, key.c_str(), json_in.size());
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}

void solidity_convertert::print_json_array_element(nlohmann::json &json_in,
    const std::string& node_type, const unsigned index)
{
  printf("### node[%u]: nodeType=%s ###\n", index, node_type.c_str());
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}
