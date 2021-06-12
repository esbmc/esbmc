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

solidity_convertert::solidity_convertert(contextt &_context, nlohmann::json &_ast_json):
  context(_context),
  ast_json(_ast_json)
{
}

bool solidity_convertert::convert()
{
  // This method convert each declarations in ast_json to symbols and add them to the context.

  // By now the context should have the symbols of all ESBMC's intrinsics and the dummy main
  // We need to convert Solidity AST nodes to the equivalent symbols and add them to the context
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
  assert(decl_class != SolidityTypes::DeclError);
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

bool solidity_convertert::get_function(jsonTrackerRef json_tracker)
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

  if(json_tracker->get_isVariadic())
    type.make_ellipsis();

  if(json_tracker->get_isInlined())
    type.inlined(true);

  locationt location_begin;
  get_location_from_decl(json_tracker, location_begin);

  std::string id, name;
  get_decl_name(json_tracker, name, id);

  symbolt symbol;
  std::string debug_modulename = get_modulename_from_path(json_tracker);
  get_default_symbol(
    symbol,
    debug_modulename,
    type,
    name,
    id,
    location_begin);

  symbol.lvalue = true;

  symbol.is_extern = json_tracker->get_storage_class() == SolidityTypes::SC_Extern ||
                     json_tracker->get_storage_class() == SolidityTypes::SC_PrivateExtern;
  symbol.file_local = (json_tracker->get_storage_class() == SolidityTypes::SC_Static);

  symbolt &added_symbol = *move_symbol_to_context(symbol);

  // Parameters: We convert the parameters first so their symbol are added to context
  // before converting the body, as they may appear on the function body
  size_t idx = 0;;
  for (; idx < json_tracker->get_num_param(); ++idx)
  {
    //json_tracker->get_function_param(idx)->print_func_param();
    /*
    code_typet::argumentt param;
    if(get_function_params(json_tracker->get_function_param(idx), param))
      return true;

    type.arguments().push_back(param);
    */
  }
  printf("@@@ number of function param decls: %lu\n", idx);

  // TODO: fd.hasBody()?
  assert(!"done?");

  return false;
}

/*
bool solidity_convertert::get_function_params(
  decl_function_tracker::funcParam* &pd,
  exprt &param)
{
  typet param_type;
}
*/

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

std::string solidity_convertert::get_modulename_from_path(jsonTrackerRef json_tracker)
{
  std::string moduleName = json_tracker->get_moduleName();
  assert(moduleName != ""); // just to be defensive
  return moduleName;
}

void solidity_convertert::get_decl_name(
    jsonTrackerRef json_tracker,
    std::string &name, std::string &id)
{
  id = name = json_tracker->get_declName();
  assert(name != "");

  switch(json_tracker->getDeclClass()) // for __ESBMC_assume, the call from get_function falls in default. the call from get_function_params falls in case ParmVar.
  {
    default:
      if(name.empty())
      {
        std::cerr << "Declaration still has an empty name\n";
        abort();
      }
  }

  id = json_tracker->get_id();
  if (id != "")
  {
    return;
  }
  else
  {
    assert(!"should not be here - get invalid id");
  }

  // Otherwise, abort
  std::cerr << "Unable to complete get_decl_name\n";
  abort();
}

void solidity_convertert::get_location_from_decl(jsonTrackerRef json_tracker, locationt &location)
{
  std::string function_name;

  if(json_tracker->get_isFunctionOrMethod())
  {
    assert(!"come back and continue - isFunctionOrMethod() returns true");
  }

  unsigned PLoc = get_presumed_location(json_tracker); // line number
  //printf("@@@ This is Ploc line number: %u\n", PLoc);

  set_location(PLoc, function_name, location); // for __ESBMC_assume, function_name is still empty after this line.
}

unsigned solidity_convertert::get_presumed_location(jsonTrackerRef json_tracker)
{
  // to keep it consistent with clang-c-frontend
  return json_tracker->get_ploc_line();
}

void solidity_convertert::set_location(unsigned PLoc, std::string &function_name, locationt &location)
{
  if (PLoc == decl_function_tracker::plocLineInvalid)
  {
    assert(!"found invalid PLoc");
    location.set_file("<invalid sloc>");
    return;
  }

  location.set_line(PLoc); // line number : unsigned signed
  location.set_file(get_filename_from_path()); // string : path + file name

  if(!function_name.empty())
    location.set_function(function_name);
}

std::string solidity_convertert::get_filename_from_path()
{
  return "Contract-Under-Test"; // TODO: just give a universal name, to be improved in the future
}

bool solidity_convertert::get_type(jsonTrackerRef json_tracker, typet &new_type)
{
  if (get_type(json_tracker->getTypeClass(), new_type, json_tracker))
    return true;

  if(json_tracker->get_isConstQualified())
    new_type.cmt_constant(true);

  if(json_tracker->get_isVolatileQualified())
    new_type.cmt_volatile(true);

  if(json_tracker->get_isRestrictQualified())
    new_type.restricted(true);

  return false;
}

bool solidity_convertert::get_type(
  const SolidityTypes::typeClass the_type,
  typet &new_type, jsonTrackerRef json_tracker)
{
  assert(the_type != SolidityTypes::TypeError); // must be a valid type class
  switch(the_type)
  {
    case SolidityTypes::TypeBuiltin:
      {
        if(get_builtin_type(json_tracker->getBuiltInType(), new_type))
          return true;
        break;
      }
    default:
      std::cerr << "Conversion of unsupported type: \"";
      std::cerr << SolidityTypes::typeClass_to_str(the_type) << std::endl;
      return true;
  }

  return false;
}

bool solidity_convertert::get_builtin_type(
  SolidityTypes::builInTypes the_blti_type,
  typet &new_type)
{
  std::string c_type;
  assert(the_blti_type != SolidityTypes::BuiltInError);
  switch(the_blti_type)
  {
    case SolidityTypes::BuiltInVoid:
      {
        new_type = empty_typet();
        c_type = "void";
        break;
      }
    case SolidityTypes::BuiltInBool:
      {
        new_type = bool_type();
        c_type = "bool";
        break;
      }
    default:
      std::cerr << "Unrecognized builtin type "
                << SolidityTypes::builInTypes_to_str(the_blti_type)
                << std::endl;
      return true;
  }

  new_type.set("#cpp_type", c_type);
  return false;
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
