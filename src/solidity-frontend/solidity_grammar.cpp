#include <solidity-frontend/solidity_grammar.h>
#include <fmt/core.h>
#include <set>
#include <util/message.h>

#define ENUM_TO_STR(s)                                                         \
  case s:                                                                      \
  {                                                                            \
    return #s;                                                                 \
  }

namespace SolidityGrammar
{
const std::unordered_map<std::string, ElementaryTypeNameT>
  uint_string_to_type_map = {
    {"uint8", UINT8},     {"uint16", UINT16},   {"uint24", UINT24},
    {"uint32", UINT32},   {"uint40", UINT40},   {"uint48", UINT48},
    {"uint56", UINT56},   {"uint64", UINT64},   {"uint72", UINT72},
    {"uint80", UINT80},   {"uint88", UINT88},   {"uint96", UINT96},
    {"uint104", UINT104}, {"uint112", UINT112}, {"uint120", UINT120},
    {"uint128", UINT128}, {"uint136", UINT136}, {"uint144", UINT144},
    {"uint152", UINT152}, {"uint160", UINT160}, {"uint168", UINT168},
    {"uint176", UINT176}, {"uint184", UINT184}, {"uint192", UINT192},
    {"uint200", UINT200}, {"uint208", UINT208}, {"uint216", UINT216},
    {"uint224", UINT224}, {"uint232", UINT232}, {"uint240", UINT240},
    {"uint248", UINT248}, {"uint256", UINT256},
};

const std::unordered_map<ElementaryTypeNameT, unsigned int> uint_size_map = {
  {UINT8, 8},     {UINT16, 16},   {UINT24, 24},   {UINT32, 32},
  {UINT40, 40},   {UINT48, 48},   {UINT56, 56},   {UINT64, 64},
  {UINT72, 72},   {UINT80, 80},   {UINT88, 88},   {UINT96, 96},
  {UINT104, 104}, {UINT112, 112}, {UINT120, 120}, {UINT128, 128},
  {UINT136, 136}, {UINT144, 144}, {UINT152, 152}, {UINT160, 160},
  {UINT168, 168}, {UINT176, 176}, {UINT184, 184}, {UINT192, 192},
  {UINT200, 200}, {UINT208, 208}, {UINT216, 216}, {UINT224, 224},
  {UINT232, 232}, {UINT240, 240}, {UINT248, 248}, {UINT256, 256},
};
const std::map<std::string, ElementaryTypeNameT> int_string_to_type_map = {
  {"int8", INT8},     {"int16", INT16},   {"int24", INT24},
  {"int32", INT32},   {"int40", INT40},   {"int48", INT48},
  {"int56", INT56},   {"int64", INT64},   {"int72", INT72},
  {"int80", INT80},   {"int88", INT88},   {"int96", INT96},
  {"int104", INT104}, {"int112", INT112}, {"int120", INT120},
  {"int128", INT128}, {"int136", INT136}, {"int144", INT144},
  {"int152", INT152}, {"int160", INT160}, {"int168", INT168},
  {"int176", INT176}, {"int184", INT184}, {"int192", INT192},
  {"int200", INT200}, {"int208", INT208}, {"int216", INT216},
  {"int224", INT224}, {"int232", INT232}, {"int240", INT240},
  {"int248", INT248}, {"int256", INT256},
};
const std::map<ElementaryTypeNameT, unsigned int> int_size_map = {
  {INT8, 8},     {INT16, 16},   {INT24, 24},   {INT32, 32},   {INT40, 40},
  {INT48, 48},   {INT56, 56},   {INT64, 64},   {INT72, 72},   {INT80, 80},
  {INT88, 88},   {INT96, 96},   {INT104, 104}, {INT112, 112}, {INT120, 120},
  {INT128, 128}, {INT136, 136}, {INT144, 144}, {INT152, 152}, {INT160, 160},
  {INT168, 168}, {INT176, 176}, {INT184, 184}, {INT192, 192}, {INT200, 200},
  {INT208, 208}, {INT216, 216}, {INT224, 224}, {INT232, 232}, {INT240, 240},
  {INT248, 248}, {INT256, 256},
};
// rule contract-body-element
ContractBodyElementT get_contract_body_element_t(const nlohmann::json &element)
{
  if(
    element["nodeType"] == "VariableDeclaration" &&
    element["stateVariable"] == true)
  {
    return StateVarDecl;
  }
  else if(
    element["nodeType"] == "FunctionDefinition" &&
    element["kind"] == "function")
  {
    return FunctionDef;
  }
  else
  {
    log_error(
      "Got contract-body-element nodeType={}. Unsupported "
      "contract-body-element type",
      element["nodeType"].get<std::string>());
    abort();
  }
  return ContractBodyElementTError;
}

const char *contract_body_element_to_str(ContractBodyElementT type)
{
  switch(type)
  {
    ENUM_TO_STR(StateVarDecl)
    ENUM_TO_STR(FunctionDef)
    ENUM_TO_STR(ContractBodyElementTError)
  default:
  {
    assert(!"Unknown contract-body-element type");
    return "UNKNOWN";
  }
  }
}

// rule type-name
TypeNameT get_type_name_t(const nlohmann::json &type_name)
{
  // Solidity AST node has duplicate descrptions: ["typeName"]["typeDescriptions"] and ["typeDescriptions"]
  if(type_name.contains("typeString"))
  {
    // for AST node that contains ["typeName"]["typeDescriptions"]
    std::string typeString = type_name["typeString"].get<std::string>();

    if(
      uint_string_to_type_map.count(typeString) ||
      int_string_to_type_map.count(typeString) || typeString == "bool" ||
      typeString == "string" || typeString.find("literal_string") == 0 ||
      typeString == "string storage ref" || typeString == "string memory")
    {
      // For state var declaration,
      return ElementaryTypeName;
    }
    else if(typeString.find("int_const") != std::string::npos)
    {
      // For Literal, their typeString is like "int_const 100".
      return ElementaryTypeName;
    }
    else if(typeString.find("function") != std::string::npos)
    {
      // FunctionToPointer decay in CallExpr when making a function call
      return Pointer;
    }
    else if(
      type_name["typeIdentifier"].get<std::string>().find("ArrayToPtr") !=
      std::string::npos)
    {
      // ArrayToPointer decay in DeclRefExpr when dereferencing an array, e.g. a[0]
      return PointerArrayToPtr;
    }
    else if(
      type_name["typeIdentifier"].get<std::string>().find("array") !=
      std::string::npos)
    {
      // Solidity's array type description is like:
      //  "typeIdentifier": "t_array$_t_uint8_$2_memory_ptr",
      //  "typeString": "uint8[2] memory"
      // Need to search for the substring "array" as in "typeIdentifier"
      if(
        type_name["typeIdentifier"].get<std::string>().find("dyn") !=
        std::string::npos)
        return DynArrayTypeName;

      return ArrayTypeName;
    }
    else
    {
      log_error(
        "Got type-name typeString={}. Unsupported type-name type",
        type_name["typeString"].get<std::string>());
      abort();
    }
  }
  else
  {
    // for AST node that contains ["typeDescriptions"] only
    if(type_name["nodeType"] == "ParameterList")
    {
      return ParameterList;
    }
    else
    {
      log_error(
        "Got type-name nodeType={}. Unsupported type-name type",
        type_name["nodeType"].get<std::string>());
      abort();
    }
  }

  return TypeNameTError; // to make some old compiler happy
}

const char *type_name_to_str(TypeNameT type)
{
  switch(type)
  {
    ENUM_TO_STR(ElementaryTypeName)
    ENUM_TO_STR(ParameterList)
    ENUM_TO_STR(Pointer)
    ENUM_TO_STR(PointerArrayToPtr)
    ENUM_TO_STR(ArrayTypeName)
    ENUM_TO_STR(DynArrayTypeName)
    ENUM_TO_STR(TypeNameTError)
  default:
  {
    assert(!"Unknown type-name type");
    return "UNKNOWN";
  }
  }
}

// rule elementary-type-name
ElementaryTypeNameT get_elementary_type_name_t(const nlohmann::json &type_name)
{
  std::string typeString = type_name["typeString"].get<std::string>();
  // rule unsigned-integer-type

  if(uint_string_to_type_map.count(typeString))
  {
    return uint_string_to_type_map.at(typeString);
  }
  if(int_string_to_type_map.count(typeString))
  {
    return int_string_to_type_map.at(typeString);
  }
  if(typeString == "bool")
  {
    return BOOL;
  }
  if(typeString.find("int_const") != std::string::npos)
  {
    /**
     * For Literal, their typeString is like "int_const 100".
     * There is no additional type info (bitsize, signed/unsigned),
     * This means it will require additional type info from the parent
     * expr to create an internal type.
     */
    return INT_LITERAL;
  }
  if(typeString.find("literal_string") == 0)
  {
    return STRING_LITERAL;
  }
  if(
    typeString == "string" || typeString == "string storage ref" ||
    typeString == "string memory")
  {
    return STRING;
  }

  log_error(
    "Got elementary-type-name typeString={}. Unsupported "
    "elementary-type-name type",
    type_name["typeString"].get<std::string>());
  abort();
}

const char *elementary_type_name_to_str(ElementaryTypeNameT type)
{
  switch(type)
  {
    ENUM_TO_STR(UINT8)
    ENUM_TO_STR(UINT16)
    ENUM_TO_STR(UINT24)
    ENUM_TO_STR(UINT32)
    ENUM_TO_STR(UINT40)
    ENUM_TO_STR(UINT48)
    ENUM_TO_STR(UINT56)
    ENUM_TO_STR(UINT64)
    ENUM_TO_STR(UINT72)
    ENUM_TO_STR(UINT80)
    ENUM_TO_STR(UINT88)
    ENUM_TO_STR(UINT96)
    ENUM_TO_STR(UINT104)
    ENUM_TO_STR(UINT112)
    ENUM_TO_STR(UINT120)
    ENUM_TO_STR(UINT128)
    ENUM_TO_STR(UINT136)
    ENUM_TO_STR(UINT144)
    ENUM_TO_STR(UINT152)
    ENUM_TO_STR(UINT160)
    ENUM_TO_STR(UINT168)
    ENUM_TO_STR(UINT176)
    ENUM_TO_STR(UINT184)
    ENUM_TO_STR(UINT192)
    ENUM_TO_STR(UINT200)
    ENUM_TO_STR(UINT208)
    ENUM_TO_STR(UINT216)
    ENUM_TO_STR(UINT224)
    ENUM_TO_STR(UINT232)
    ENUM_TO_STR(UINT240)
    ENUM_TO_STR(UINT248)
    ENUM_TO_STR(UINT256)
    ENUM_TO_STR(INT_LITERAL)
    ENUM_TO_STR(INT8)
    ENUM_TO_STR(INT16)
    ENUM_TO_STR(INT24)
    ENUM_TO_STR(INT32)
    ENUM_TO_STR(INT40)
    ENUM_TO_STR(INT48)
    ENUM_TO_STR(INT56)
    ENUM_TO_STR(INT64)
    ENUM_TO_STR(INT72)
    ENUM_TO_STR(INT80)
    ENUM_TO_STR(INT88)
    ENUM_TO_STR(INT96)
    ENUM_TO_STR(INT104)
    ENUM_TO_STR(INT112)
    ENUM_TO_STR(INT120)
    ENUM_TO_STR(INT128)
    ENUM_TO_STR(INT136)
    ENUM_TO_STR(INT144)
    ENUM_TO_STR(INT152)
    ENUM_TO_STR(INT160)
    ENUM_TO_STR(INT168)
    ENUM_TO_STR(INT176)
    ENUM_TO_STR(INT184)
    ENUM_TO_STR(INT192)
    ENUM_TO_STR(INT200)
    ENUM_TO_STR(INT208)
    ENUM_TO_STR(INT216)
    ENUM_TO_STR(INT224)
    ENUM_TO_STR(INT232)
    ENUM_TO_STR(INT240)
    ENUM_TO_STR(INT248)
    ENUM_TO_STR(INT256)
    ENUM_TO_STR(BOOL)
    ENUM_TO_STR(STRING_LITERAL)
    ENUM_TO_STR(STRING)
    ENUM_TO_STR(ElementaryTypeNameTError)
  default:
  {
    assert(!"Unknown elementary-type-name type");
    return "UNKNOWN";
  }
  }
}

unsigned int uint_type_name_to_size(ElementaryTypeNameT type)
{
  return uint_size_map.at(type);
}

unsigned int int_type_name_to_size(ElementaryTypeNameT type)
{
  return int_size_map.at(type);
}

// rule parameter-list
ParameterListT get_parameter_list_t(const nlohmann::json &type_name)
{
  if(type_name["parameters"].size() == 0)
  {
    return EMPTY;
  }
  else
  {
    return NONEMPTY;
  }

  return ParameterListTError; // to make some old gcc compilers happy
}

const char *parameter_list_to_str(ParameterListT type)
{
  switch(type)
  {
    ENUM_TO_STR(EMPTY)
    ENUM_TO_STR(NONEMPTY)
    ENUM_TO_STR(ParameterListTError)
  default:
  {
    assert(!"Unknown parameter-list type");
    return "UNKNOWN";
  }
  }
}

// rule block
BlockT get_block_t(const nlohmann::json &block)
{
  if(block["nodeType"] == "Block" && block.contains("statements"))
  {
    return Statement;
  }
  else
  {
    log_error(
      "Got block nodeType={}. Unsupported block type",
      block["nodeType"].get<std::string>());
    abort();
  }
  return BlockTError;
}

const char *block_to_str(BlockT type)
{
  switch(type)
  {
    ENUM_TO_STR(Statement)
    ENUM_TO_STR(UncheckedBlock)
    ENUM_TO_STR(BlockTError)
  default:
  {
    assert(!"Unknown block type");
    return "UNKNOWN";
  }
  }
}

// rule statement
StatementT get_statement_t(const nlohmann::json &stmt)
{
  if(stmt["nodeType"] == "ExpressionStatement")
  {
    return ExpressionStatement;
  }
  else if(stmt["nodeType"] == "VariableDeclarationStatement")
  {
    return VariableDeclStatement;
  }
  else if(stmt["nodeType"] == "Return")
  {
    return ReturnStatement;
  }
  else if(stmt["nodeType"] == "ForStatement")
  {
    return ForStatement;
  }
  else if(stmt["nodeType"] == "Block")
  {
    return Block;
  }
  else if(stmt["nodeType"] == "IfStatement")
  {
    return IfStatement;
  }
  else if(stmt["nodeType"] == "WhileStatement")
  {
    return WhileStatement;
  }
  else
  {
    log_error(
      "Got statement nodeType={}. Unsupported statement type",
      stmt["nodeType"].get<std::string>());
    abort();
  }
  return StatementTError;
}

const char *statement_to_str(StatementT type)
{
  switch(type)
  {
    ENUM_TO_STR(Block)
    ENUM_TO_STR(ExpressionStatement)
    ENUM_TO_STR(VariableDeclStatement)
    ENUM_TO_STR(ReturnStatement)
    ENUM_TO_STR(ForStatement)
    ENUM_TO_STR(IfStatement)
    ENUM_TO_STR(WhileStatement)
    ENUM_TO_STR(StatementTError)
  default:
  {
    assert(!"Unknown statement type");
    return "UNKNOWN";
  }
  }
}

// rule expression
ExpressionT get_expression_t(const nlohmann::json &expr)
{
  if(expr["nodeType"] == "Assignment" || expr["nodeType"] == "BinaryOperation")
  {
    return BinaryOperatorClass;
  }
  else if(expr["nodeType"] == "UnaryOperation")
  {
    return UnaryOperatorClass;
  }
  else if(
    expr["nodeType"] == "Identifier" && expr.contains("referencedDeclaration"))
  {
    return DeclRefExprClass;
  }
  else if(expr["nodeType"] == "Literal")
  {
    return Literal;
  }
  else if(expr["nodeType"] == "FunctionCall")
  {
    return CallExprClass;
  }
  else if(expr["nodeType"] == "ImplicitCastExprClass")
  {
    return ImplicitCastExprClass;
  }
  else if(expr["nodeType"] == "IndexAccess")
  {
    return IndexAccess;
  }
  else
  {
    log_error(
      "Got expression nodeType={}. Unsupported expression type",
      expr["nodeType"].get<std::string>());
    abort();
  }
  return ExpressionTError;
}

ExpressionT get_unary_expr_operator_t(const nlohmann::json &expr, bool uo_pre)
{
  if(expr["operator"] == "--")
  {
    if(uo_pre)
      return UO_PreDec;

    assert(!"Unsupported - UO_PostDec");
  }
  if(expr["operator"] == "++")
  {
    if(uo_pre)
      return UO_PreInc;

    assert(!"Unsupported - UO_PostDec");
  }
  if(expr["operator"] == "-")
  {
    return UO_Minus;
  }
  log_error(
    "Got expression operator={}. Unsupported expression operator",
    expr["operator"].get<std::string>());

  abort();
}

ExpressionT get_expr_operator_t(const nlohmann::json &expr)
{
  if(expr["operator"] == "=")
  {
    return BO_Assign;
  }
  else if(expr["operator"] == "+")
  {
    return BO_Add;
  }
  else if(expr["operator"] == "-")
  {
    return BO_Sub;
  }
  else if(expr["operator"] == ">")
  {
    return BO_GT;
  }
  else if(expr["operator"] == "<")
  {
    return BO_LT;
  }
  else if(expr["operator"] == "!=")
  {
    return BO_NE;
  }
  else if(expr["operator"] == "==")
  {
    return BO_EQ;
  }
  else if(expr["operator"] == "%")
  {
    return BO_Rem;
  }
  else if(expr["operator"] == "&&")
  {
    return BO_LAnd;
  }
  else
  {
    log_error(
      "Got expression operator={}. Unsupported expression operator",
      expr["operator"].get<std::string>());
    abort();
  }

  return ExpressionTError; // make some old compilers happy
}

const char *expression_to_str(ExpressionT type)
{
  switch(type)
  {
    ENUM_TO_STR(BinaryOperatorClass)
    ENUM_TO_STR(BO_Assign)
    ENUM_TO_STR(BO_Add)
    ENUM_TO_STR(BO_Sub)
    ENUM_TO_STR(BO_GT)
    ENUM_TO_STR(BO_LT)
    ENUM_TO_STR(BO_NE)
    ENUM_TO_STR(BO_EQ)
    ENUM_TO_STR(BO_Rem)
    ENUM_TO_STR(BO_LAnd)
    ENUM_TO_STR(UnaryOperatorClass)
    ENUM_TO_STR(UO_PreDec)
    ENUM_TO_STR(UO_PreInc)
    ENUM_TO_STR(UO_Minus)
    ENUM_TO_STR(DeclRefExprClass)
    ENUM_TO_STR(Literal)
    ENUM_TO_STR(CallExprClass)
    ENUM_TO_STR(ImplicitCastExprClass)
    ENUM_TO_STR(IndexAccess)
    ENUM_TO_STR(ExpressionTError)
  default:
  {
    assert(!"Unknown expression type");
    return "UNKNOWN";
  }
  }
}

// rule variable-declaration-statement
VarDeclStmtT get_var_decl_stmt_t(const nlohmann::json &stmt)
{
  if(
    stmt["nodeType"] == "VariableDeclaration" && stmt["stateVariable"] == false)
  {
    return VariableDecl;
  }
  else
  {
    log_error(
      "Got expression nodeType={}. Unsupported "
      "variable-declaration-statement operator",
      stmt["nodeType"].get<std::string>());
    abort();
  }
  return VarDeclStmtTError; // make some old compilers happy
}

const char *var_decl_statement_to_str(VarDeclStmtT type)
{
  switch(type)
  {
    ENUM_TO_STR(VariableDecl)
    ENUM_TO_STR(VariableDeclTuple)
    ENUM_TO_STR(VarDeclStmtTError)
  default:
  {
    assert(!"Unknown variable-declaration-statement type");
    return "UNKNOWN";
  }
  }
}

// auxiliary type to convert function call
FunctionDeclRefT get_func_decl_ref_t(const nlohmann::json &decl)
{
  assert(decl["nodeType"] == "FunctionDefinition");
  if(decl["parameters"]["parameters"].size() == 0)
  {
    return FunctionNoProto;
  }
  else
  {
    return FunctionProto;
  }
  return FunctionDeclRefTError; // to make some old compilers happy
}

const char *func_decl_ref_to_str(FunctionDeclRefT type)
{
  switch(type)
  {
    ENUM_TO_STR(FunctionProto)
    ENUM_TO_STR(FunctionNoProto)
    ENUM_TO_STR(FunctionDeclRefTError)
  default:
  {
    assert(!"Unknown auxiliary type to convert function call");
    return "UNKNOWN";
  }
  }
}

// auxiliary type for implicit casting
ImplicitCastTypeT get_implicit_cast_type_t(std::string cast)
{
  if(cast == "LValueToRValue")
  {
    return LValueToRValue;
  }
  else if(cast == "FunctionToPointerDecay")
  {
    return FunctionToPointerDecay;
  }
  else if(cast == "ArrayToPointerDecay")
  {
    return ArrayToPointerDecay;
  }
  else
  {
    log_error("Got implicit cast type={}. Unsupported case type", cast.c_str());
    abort();
  }

  return ImplicitCastTypeTError; // to make some old compilers happy
}

const char *implicit_cast_type_to_str(ImplicitCastTypeT type)
{
  switch(type)
  {
    ENUM_TO_STR(LValueToRValue)
    ENUM_TO_STR(FunctionToPointerDecay)
    ENUM_TO_STR(ArrayToPointerDecay)
    ENUM_TO_STR(ImplicitCastTypeTError)
  default:
  {
    assert(!"Unknown auxiliary type for implicit casting");
    return "UNKNOWN";
  }
  }
}
}; // namespace SolidityGrammar
