#include <solidity-ast-frontend/solidity_decl_tracker.h>

void decl_function_tracker::config()
{
  //TODO: these configurations were created heavily influenced by clang-frontend.
  //      Some of them may be redundant for Solidity. Future audit might be needed.

  set_isImplicit();
  set_isDefined();
  set_isThisDeclarationADefinition();
  set_type_class();
  set_builtin_type();
  set_isConstQualified();
  set_isVolatileQualified();
  set_isRestrictQualified();
  set_isVariadic();
  set_isInlined();
  set_isFunctionOrMethod();
  set_ploc_line();
  set_declName();
  set_id();
  set_moduleName();
  set_storageClass();
  set_num_param();
  // deal with params of this function declaration
  if (num_param)
    populate_params();
}

void decl_function_tracker::set_moduleName()
{
  assert(moduleName == ""); // only allowed to set once during config(). If set twice, something is wrong.
  if (!decl_func.contains("moduleName"))
    assert(!"missing \'moduleName\' in DeclFunction");
  moduleName = decl_func["moduleName"].get<std::string>();
}

void decl_function_tracker::set_id()
{
  assert(id == ""); // only allowed to set once during config(). If set twice, something is wrong.
  if (!decl_func.contains("id"))
    assert(!"missing \'id\' in DeclFunction");
  id = decl_func["id"].get<std::string>();
}

void decl_function_tracker::set_declName()
{
  assert(declName == ""); // only allowed to set once during config(). If set twice, something is wrong.
  if (!decl_func.contains("declName"))
    assert(!"missing \'declName\' in DeclFunction");
  declName = decl_func["declName"].get<std::string>();
}

void decl_function_tracker::set_ploc_line()
{
  assert(PLoc_Line == plocLineInvalid); // only allowed to set once during config(). If set twice, something is wrong.
  if (!decl_func.contains("PLoc_Line"))
    assert(!"missing \'PLoc_Line\' in DeclFunction");
  PLoc_Line = decl_func["PLoc_Line"].get<unsigned>();
}

void decl_function_tracker::set_isFunctionOrMethod()
{
  if (!decl_func.contains("isFunctionOrMethod"))
    assert(!"missing \'isFunctionOrMethod\' in DeclFunction");
  isFunctionOrMethod = (decl_func["isFunctionOrMethod"].get<bool>())? true : false;
}

void decl_function_tracker::set_isVariadic()
{
  if (!decl_func.contains("isVariadic"))
    assert(!"missing \'isVariadic\' in DeclFunction");
  isVariadic = (decl_func["isVariadic"].get<bool>())? true : false;
}

void decl_function_tracker::set_isInlined()
{
  if (!decl_func.contains("isInlined"))
    assert(!"missing \'isInlined\' in DeclFunction");
  isInlined = (decl_func["isInlined"].get<bool>())? true : false;
}

void decl_function_tracker::set_isConstQualified()
{
  if (!decl_func.contains("isConstQualified"))
    assert(!"missing \'isConstQualified\' in DeclFunction");
  isConstQualified = (decl_func["isConstQualified"].get<bool>())? true : false;
}

void decl_function_tracker::set_isVolatileQualified()
{
  if (!decl_func.contains("isVolatileQualified"))
    assert(!"missing \'isVolatileQualified\' in DeclFunction");
  isVolatileQualified = (decl_func["isVolatileQualified"].get<bool>())? true : false;
}

void decl_function_tracker::set_isRestrictQualified()
{
  if (!decl_func.contains("isRestrictQualified"))
    assert(!"missing \'isRestrictQualified\' in DeclFunction");
  isRestrictQualified = (decl_func["isRestrictQualified"].get<bool>())? true : false;
}

void decl_function_tracker::set_isImplicit()
{
  if (!decl_func.contains("isImplicit"))
    assert(!"missing \'isImplicit\' in DeclFunction");
  isImplicit = (decl_func["isImplicit"].get<bool>())? true : false;
}

void decl_function_tracker::set_isDefined()
{
  if (!decl_func.contains("isDefined"))
    assert(!"missing \'isDefined\' in DeclFunction");
  isDefined = (decl_func["isDefined"].get<bool>())? true : false;
}

void decl_function_tracker::set_isThisDeclarationADefinition()
{
  if (!decl_func.contains("isThisDeclarationADefinition"))
    assert(!"missing \'isThisDeclarationADefinition\' in DeclFunction");
  isThisDeclarationADefinition = (decl_func["isThisDeclarationADefinition"].get<bool>())? true : false;
}

void decl_function_tracker::set_type_class()
{
  assert(type_class == SolidityTypes::TypeError); // only allowed to set once during config(). If set twice, something is wrong.
  if (!decl_func.contains("typeClass"))
    assert(!"missing \'typeClass\' in DeclFunction");
  type_class = SolidityTypes::get_type_class(
      decl_func["typeClass"].get<std::string>()
      );
}

void decl_function_tracker::set_decl_class()
{
  assert(decl_class == SolidityTypes::DeclError); // only allowed to set once during config(). If set twice, something is wrong.
  if (!decl_func.contains("declClass"))
    assert(!"missing \'declClass\' in DeclFunction");
  decl_class = SolidityTypes::get_decl_class(
      decl_func["declClass"].get<std::string>()
      );
}

void decl_function_tracker::set_builtin_type()
{
  assert(builtin_type == SolidityTypes::BuiltInError); // only allowed to set once during config(). If set twice, something is wrong.
  if (!decl_func.contains("builtInTypes"))
    assert(!"missing \'builtInTypes\' in DeclFunction");
  builtin_type = SolidityTypes::get_builtin_type(
      decl_func["builtInTypes"].get<std::string>()
      );
}

void decl_function_tracker::set_storageClass()
{
  assert(storage_class == SolidityTypes::SCError); // only allowed to set once during config(). If set twice, something is wrong.
  if (!decl_func.contains("storageClass"))
    assert(!"missing \'storageClass\' in DeclFunction");
  storage_class = SolidityTypes::get_storage_class(
      decl_func["storageClass"].get<std::string>()
      );
}

void decl_function_tracker::set_num_param()
{
  assert(num_param == numParamInvalid); // only allowed to set once during config(). If set twice, something is wrong.
  if (!decl_func.contains("parameters"))
    assert(!"missing \'parameters\' in DeclFunction");
  num_param = decl_func["parameters"].size();
}

void decl_function_tracker::populate_params()
{
  size_t param_array_size = decl_func["parameters"].size();
  assert(params.size() == 0); // only allowed to set once during config(). If set twice, something is wrong.
  assert(param_array_size == num_param); // to be more defensive
  for (size_t i = 0; i < param_array_size; ++i)
  {
    nlohmann::json& tmp_param = decl_func["parameters"].at(i);
    //std::cout << std::setw(2) << tmp_param << '\n'; // debug print of tmp_param

    funcParam param;
    param.type_class = SolidityTypes::get_type_class(tmp_param["typeClass"].get<std::string>());
    param.decl_class = SolidityTypes::get_decl_class(tmp_param["declClass"].get<std::string>());
    param.builtin_type = SolidityTypes::get_builtin_type(tmp_param["builtInTypes"].get<std::string>());
    param.isConstQualified = (tmp_param["isConstQualified"].get<bool>())? true : false;
    param.isVolatileQualified = (tmp_param["isVolatileQualified"].get<bool>())? true : false;
    param.isRestrictQualified = (tmp_param["isRestrictQualified"].get<bool>())? true : false;
    param.isArray = (tmp_param["isArray"].get<bool>())? true : false;
    param.nameEmpty = (tmp_param["nameEmpty"].get<bool>())? true : false;

    params.push_back(param); // copy constructor of funcParam is getting called here
  }

  assert(params.size() == num_param); // param size must match
}

void decl_function_tracker::print_decl_func_json()
{
  const nlohmann::json &json_in = decl_func;
  printf("### decl_func_json: ###\n");
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}

