#include <solidity-ast-frontend/solidity_convert.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <iomanip>

solidity_convertert::solidity_convertert(contextt &_context, nlohmann::json &_ast_json, nlohmann::json &_intrinsic_json):
  context(_context),
  ast_json(_ast_json),
  intrinsic_json(_intrinsic_json)
{
}

bool solidity_convertert::convert()
{
  // This method convert each declarations in ast_json to symbols and add them to the context.
  // The conversion consists of three parts:
  //  - PART 1: get declarations from intrinsics include ESBMC and TACAS definitions
  //  - PART 2: get declarations from AST nodes

  // PART 1: get declarations from intrinsics include ESBMC and TACAS definitions
  unsigned index = 0;
  for (auto it = intrinsic_json.begin(); it != intrinsic_json.end(); ++it, ++index)
  {
    // iterate over each json object (i.e. ESBMC or TACAS definitions) and add symbols as we go
    const std::string& top_level_key = it.key();
    printf("@@ converting %s ... \n", top_level_key.c_str());
    exprt dummy_decl;
    get_decl_intrinsics(it.value(), dummy_decl, index, top_level_key, "intrinsic_json");
  }

  //TODO:  - PART 2: get declarations from AST nodes
  assert(!"come back and continue - PART 2 conversion");

  return false;
}

bool solidity_convertert::get_decl_intrinsics(
    nlohmann::json& decl, exprt &new_expr,
    const unsigned index, const std::string &key, const std::string &json_name)
{
  // This method convert declarations from intrinsics including ESBMC and TACAS definitions.
  // It's called when those declarations are to be added to the context.
  //print_json_element(decl, index, key, json_name);

  if (!decl.contains("declClass"))
  {
    printf("missing \'declClass\' key in %s[%u]: %s\n", json_name.c_str(), index, key.c_str());
    assert(0);
  }

  // First we need to get Decl class before making a json tracker
  SolidityTypes::declClass decl_class =
    SolidityTypes::get_decl_class(static_cast<std::string>(decl.at("declClass")));
  switch(decl_class)
  {
    // Declaration of functions
    case SolidityTypes::DeclFunction:
    {
      // make a tracker from the json object to facilitate our symbol conversion
      // avoid using a global tracker as a member within this class for performance reasons
      auto json_tracker = std::make_shared<decl_function_tracker>(decl);
      json_tracker->config();
      get_function(json_tracker);

      assert(!"processing DeclFunction ...\n");
      break;
    }
    default:
      std::cerr << "**** ERROR: ";
      std::cerr << "Unrecognized / unimplemented declaration "
                << decl.at("declClass") << std::endl;
      return true; // 'true' indicates something is wrong
  }

  return false; // 'false' indicates everything is fine
}

bool solidity_convertert::get_function(std::shared_ptr<decl_function_tracker>& json_tracker)
{
  if (json_tracker->get_isImplicit())
    return false;

  if (json_tracker->get_isDefined() &&
      !json_tracker->get_isThisDeclarationADefinition())
    return false;

  // need equivalent for old_functionDecl and current_scope_var_num?

  // Build function's type
  code_typet type;

  // Return type
  if(get_type(json_tracker, type.return_type()))
    return true;


  assert(!"done?");

  return false;
}

bool solidity_convertert::get_type(
    std::shared_ptr<decl_function_tracker>& json_tracker,
    typet &new_type)
{
  if (get_type(json_tracker->getTypeClass(), new_type))
    return true;

  assert(!"continue - q_type");
  return false;
}

bool solidity_convertert::get_type(const SolidityTypes::typeClass the_type, typet &new_type)
{
  assert(the_type != SolidityTypes::TypeError); // must be a valid type class
  switch(the_type)
  {
    case SolidityTypes::TypeBuiltin:
      {
        assert(!"got type Builtin");
        break;
      }
    default:
      std::cerr << "Conversion of unsupported type: \"";
      std::cerr << SolidityTypes::typeClass_to_str(the_type) << std::endl;
      return true;
  }

  return false;
}

void solidity_convertert::print_json_element(nlohmann::json &json_in, const unsigned index,
    const std::string &key, const std::string& json_name)
{
  printf("### %s element[%u] content: key=%s, size=%lu ###\n",
      json_name.c_str(), index, key.c_str(), json_in.size());
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}
