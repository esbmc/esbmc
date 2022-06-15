/*******************************************************************\

Module: Solidity Language module

Author: Kunjian Song, kunjian.song@postgrad.manchester.ac.uk

\*******************************************************************/

#include <solidity-frontend/solidity_grammar.h>
#include <fmt/core.h>
#include <set>

#define ENUM_TO_STR(s)                                                         \
  case s:                                                                      \
  {                                                                            \
    return #s;                                                                 \
  }

namespace SolidityGrammar
{
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
    assert(!((fmt::format(
                "Got contract-body-element nodeType={}. Unsupported "
                "contract-body-element type",
                element["nodeType"].get<std::string>()))
               .c_str()));
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
    std::set<std::string> uintTypeStringSet = {
      "uint8",   "uint16",  "uint24",  "uint32",  "uint40",  "uint48",
      "uint56",  "uint64",  "uint72",  "uint80",  "uint88",  "uint96",
      "uint104", "uint112", "uint120", "uint128", "uint136", "uint144",
      "uint152", "uint160", "uint168", "uint176", "uint184", "uint192",
      "uint200", "uint208", "uint216", "uint224", "uint232", "uint240",
      "uint248", "uint256",
    };

    std::string typeString = type_name["typeString"].get<std::string>();

    if(uintTypeStringSet.count(typeString) || typeString == "bool")
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
      assert(!((fmt::format(
                  "Got type-name typeString={}. Unsupported type-name type",
                  type_name["typeString"].get<std::string>()))
                 .c_str()));
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
      assert(!((fmt::format(
                  "Got type-name nodeType={}. Unsupported type-name type",
                  type_name["nodeType"].get<std::string>()))
                 .c_str()));
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
  std::map<std::string, ElementaryTypeNameT> uintMap = {
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
  if(typeString.substr(0, 4) == "uint" && uintMap.count(typeString))
  {
    return uintMap[typeString];
  }
  else if(typeString == "bool")
  {
    return BOOL;
  }
  else if(typeString.find("int_const") != std::string::npos)
  {
    // For Literal, their typeString is like "int_const 100".
    // TODO: Fix me! For simplicity, we assume everything is unsigned int.
    return UINT256;
  }
  else
  {
    assert(!((fmt::format(
                "Got elementary-type-name typeString={}. Unsupported "
                "elementary-type-name type",
                type_name["typeString"].get<std::string>()))
               .c_str()));
  }
  return ElementaryTypeNameTError;
}

const char *elementary_type_name_to_str(ElementaryTypeNameT type)
{
  switch(type)
  {
    ENUM_TO_STR(UINT8)
    ENUM_TO_STR(BOOL)
    ENUM_TO_STR(ElementaryTypeNameTError)
  default:
  {
    assert(!"Unknown elementary-type-name type");
    return "UNKNOWN";
  }
  }
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
    assert(!((fmt::format(
                "Got block nodeType={}. Unsupported block type",
                block["nodeType"].get<std::string>()))
               .c_str()));
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
    assert(!((fmt::format(
                "Got statement nodeType={}. Unsupported statement type",
                stmt["nodeType"].get<std::string>()))
               .c_str()));
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
    assert(!((fmt::format(
                "Got expression nodeType={}. Unsupported expression type",
                expr["nodeType"].get<std::string>()))
               .c_str()));
  }
  return ExpressionTError;
}

ExpressionT get_expr_operator_t(const nlohmann::json &expr, bool uo_pre)
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
  else if(expr["operator"] == "--")
  {
    if(uo_pre)
    {
      return UO_PreDec;
    }
    else
    {
      assert(!"Unsupported - UO_PostDec");
    }
  }
  else if(expr["operator"] == "++")
  {
    if(uo_pre)
    {
      return UO_PreInc;
    }
    else
    {
      assert(!"Unsupported - UO_PostDec");
    }
  }
  else
  {
    assert(!((fmt::format(
                "Got expression operator={}. Unsupported expression operator",
                expr["operator"].get<std::string>()))
               .c_str()));
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
    ENUM_TO_STR(UnaryOperatorClass)
    ENUM_TO_STR(UO_PreDec)
    ENUM_TO_STR(UO_PreInc)
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
    assert(!((fmt::format(
                "Got expression nodeType={}. Unsupported "
                "variable-declaration-statement operator",
                stmt["nodeType"].get<std::string>()))
               .c_str()));
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
    assert(
      !((fmt::format(
           "Got implicit cast type={}. Unsupported case type", cast.c_str()))
          .c_str()));
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
