/// \file solidity_grammar.cpp
/// \brief Implementation of Solidity grammar classification functions.
///
/// Maps solc JSON AST node fields (type strings, operator tokens, node kinds)
/// to the corresponding SolidityGrammar enums. Includes lookup tables for all
/// uint/int bit-widths, operator precedence, and string-to-enum conversions.

#include <fmt/core.h>
#include <solidity-frontend/solidity_convert.h>
#include <solidity-frontend/solidity_grammar.h>
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
const std::unordered_map<std::string, ElementaryTypeNameT> bytesn_to_type_map =
  {
    {"byte", BYTES1},     {"bytes1", BYTES1},   {"bytes2", BYTES2},
    {"bytes3", BYTES3},   {"bytes4", BYTES4},   {"bytes5", BYTES5},
    {"bytes6", BYTES6},   {"bytes7", BYTES7},   {"bytes8", BYTES8},
    {"bytes9", BYTES9},   {"bytes10", BYTES10}, {"bytes11", BYTES11},
    {"bytes12", BYTES12}, {"bytes13", BYTES13}, {"bytes14", BYTES14},
    {"bytes15", BYTES15}, {"bytes16", BYTES16}, {"bytes17", BYTES17},
    {"bytes18", BYTES18}, {"bytes19", BYTES19}, {"bytes20", BYTES20},
    {"bytes21", BYTES21}, {"bytes22", BYTES22}, {"bytes23", BYTES23},
    {"bytes24", BYTES24}, {"bytes25", BYTES25}, {"bytes26", BYTES26},
    {"bytes27", BYTES27}, {"bytes28", BYTES28}, {"bytes29", BYTES29},
    {"bytes30", BYTES30}, {"bytes31", BYTES31}, {"bytes32", BYTES32},
};
const std::map<ElementaryTypeNameT, unsigned int> bytesn_size_map = {
  {BYTES1, 1},   {BYTES2, 2},   {BYTES3, 3},   {BYTES4, 4},   {BYTES5, 5},
  {BYTES6, 6},   {BYTES7, 7},   {BYTES8, 8},   {BYTES9, 9},   {BYTES10, 10},
  {BYTES11, 11}, {BYTES12, 12}, {BYTES13, 13}, {BYTES14, 14}, {BYTES15, 15},
  {BYTES16, 16}, {BYTES17, 17}, {BYTES18, 18}, {BYTES19, 19}, {BYTES20, 20},
  {BYTES21, 21}, {BYTES22, 22}, {BYTES23, 23}, {BYTES24, 24}, {BYTES25, 25},
  {BYTES26, 26}, {BYTES27, 27}, {BYTES28, 28}, {BYTES29, 29}, {BYTES30, 30},
  {BYTES31, 31}, {BYTES32, 32},
};

// rule contract-body-element
ContractBodyElementT get_contract_body_element_t(const nlohmann::json &element)
{
  if (element["nodeType"] == "VariableDeclaration")
  {
    return VarDecl;
  }
  else if (
    element["nodeType"] == "FunctionDefinition" &&
    (element["kind"] == "function" || element["kind"] == "constructor" ||
     element["kind"] == "receive" || element["kind"] == "fallback"))
  {
    return FunctionDef;
  }
  else if (element["nodeType"] == "StructDefinition")
  {
    return StructDef;
  }
  else if (element["nodeType"] == "EnumDefinition")
  {
    return EnumDef;
  }
  else if (element["nodeType"] == "ErrorDefinition")
  {
    return ErrorDef;
  }
  else if (element["nodeType"] == "EventDefinition")
  {
    return EventDef;
  }
  else if (element["nodeType"] == "UsingForDirective")
  {
    return UsingForDef;
  }
  else if (element["nodeType"] == "ModifierDefinition")
  {
    return ModifierDef;
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
  switch (type)
  {
    ENUM_TO_STR(VarDecl)
    ENUM_TO_STR(FunctionDef)
    ENUM_TO_STR(StructDef)
    ENUM_TO_STR(EnumDef)
    ENUM_TO_STR(ErrorDef)
    ENUM_TO_STR(EventDef)
    ENUM_TO_STR(UsingForDef)
    ENUM_TO_STR(ModifierDef)
    ENUM_TO_STR(ContractBodyElementTError)
  default:
  {
    assert(!"Unknown contract-body-element type");
    return "UNKNOWN";
  }
  }
}

// check if it's <address>.member
bool is_address_member_call(const nlohmann::json &expr)
{
  if (expr["nodeType"] != "MemberAccess" || !expr.contains("expression"))
    return false;
  SolidityGrammar::TypeNameT type_name =
    get_type_name_t(expr["expression"]["typeDescriptions"]);

  if (
    (type_name == SolidityGrammar::TypeNameT::AddressPayableTypeName ||
     type_name == SolidityGrammar::TypeNameT::AddressTypeName) &&
    expr.contains("memberName") &&
    (solidity_convertert::is_low_level_call(expr["memberName"]) ||
     solidity_convertert::is_low_level_property(expr["memberName"])))
    return true;

  return false;
}

// check if it is a function defined in a library
bool is_sol_library_function(const int ref_id)
{
  const auto &func_ref = solidity_convertert::find_node_by_id(
    solidity_convertert::src_ast_json["nodes"], ref_id);
  if (func_ref.empty() || func_ref.is_null())
    return false;
  if (!(func_ref["nodeType"] == "FunctionDefinition" ||
        func_ref["nodeType"] == "EventDefinition" ||
        func_ref["nodeType"] == "ErrorDefinition"))
    return false;
  const auto &lib_ref = solidity_convertert::find_parent_contract(
    solidity_convertert::src_ast_json["nodes"], func_ref);
  if (lib_ref.contains("contractKind") && lib_ref["contractKind"] == "library")
    return true;

  return false;
}

// rule type-name
TypeNameT get_type_name_t(const nlohmann::json &type_name)
{
  // Solidity AST node has duplicate descrptions: ["typeName"]["typeDescriptions"] and ["typeDescriptions"]
  //! Order matters

  if (type_name.contains("typeString"))
  {
    // for AST node that contains ["typeName"]["typeDescriptions"]
    const std::string typeString = type_name["typeString"].get<std::string>();
    const std::string typeIdentifier =
      type_name["typeIdentifier"].get<std::string>();

    // we must first handle tuple
    // otherwise we might parse tuple(literal_string, literal_string)
    // as ElementaryTypeName
    if (typeString.compare(0, 5, "type(") == 0)
    {
      if (typeIdentifier.compare(0, 17, "t_magic_meta_type") == 0)
        return TypeProperty;
      // For type conversion
      return TypeConversionName;
    }
    else if (typeString.compare(0, 6, "tuple(") == 0)
    {
      return TupleTypeName;
    }
    else if (typeIdentifier.compare(0, 10, "t_mapping$") == 0)
    {
      return MappingTypeName;
    }
    else if (typeIdentifier.compare(0, 8, "t_array$") == 0)
    {
      // Solidity's array type description is like:
      //  "typeIdentifier": "t_array$_t_uint8_$2_memory_ptr",
      //  "typeString": "uint8[2] memory"

      // The Arrays in Solidity can be classified into the following two types based on size –
      //   Fixed Size Array
      //   Dynamic Array
      // Furthermore, the solidity array can also be categorized based on where they are stored as –
      //   Storage Array
      //   Memory Array

      // Multi-Dimensional Arrays
      if (typeIdentifier.compare(0, 17, "t_array$_t_array$") == 0)
      {
        log_debug(
          "solidity", "Experimental support for multi-dimensional arrays.");
        return NestedArrayTypeName;
      }

      if (typeIdentifier.find("$dyn") != std::string::npos)
        return DynArrayTypeName;

      return ArrayTypeName;
    }
    else if (typeIdentifier.compare(0, 9, "t_struct$") == 0)
    {
      return StructTypeName;
    }
    else if (typeIdentifier.compare(0, 9, "t_address") == 0)
    {
      if (typeIdentifier.compare(0, 17, "t_address_payable") == 0)
        return AddressPayableTypeName;
      return AddressTypeName;
    }
    else if (
      uint_string_to_type_map.count(typeString) ||
      int_string_to_type_map.count(typeString) || typeString == "bool" ||
      typeString == "string" || typeString.find("literal_string") == 0 ||
      typeString == "string storage ref" ||
      typeString == "string storage pointer" || typeString == "string memory" ||
      typeString == "address payable" || typeString == "address" ||
      typeString.compare(0, 5, "bytes") == 0)
    {
      // For state var declaration,
      return ElementaryTypeName;
    }
    else if (typeIdentifier.find("t_enum$") != std::string::npos)
    {
      return EnumTypeName;
    }
    else if (typeIdentifier.compare(0, 11, "t_contract$") == 0)
    {
      return ContractTypeName;
    }

    else if (typeString.find("int_const") != std::string::npos)
    {
      // For Literal, their typeString is like "int_const 100".
      return ElementaryTypeName;
    }
    else if (
      typeString.find("function") != std::string::npos &&
      typeString.find("contract ") == std::string::npos)
    {
      // FunctionToPointer decay in CallExpr when making a function call
      return Pointer;
    }
    else if (typeIdentifier.find("ArrayToPtr") != std::string::npos)
    {
      // ArrayToPointer decay in DeclRefExpr when dereferencing an array, e.g. a[0]
      return PointerArrayToPtr;
    }
    // for Special Variables and Functions
    else if (typeIdentifier.compare(0, 7, "t_magic") == 0)
    {
      return BuiltinTypeName;
    }
    else if (typeIdentifier.compare(0, 7, "t_error") == 0)
    {
      return ErrorTypeName;
    }
    else if (solidity_convertert::UserDefinedVarMap.count(typeString) > 0)
    {
      return UserDefinedTypeName;
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
    // for AST node that does not contain ["typeDescriptions"] only
    // function returnParameters
    if (type_name["nodeType"] == "ParameterList")
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
  switch (type)
  {
    ENUM_TO_STR(ElementaryTypeName)
    ENUM_TO_STR(ParameterList)
    ENUM_TO_STR(Pointer)
    ENUM_TO_STR(PointerArrayToPtr)
    ENUM_TO_STR(ArrayTypeName)
    ENUM_TO_STR(DynArrayTypeName)
    ENUM_TO_STR(NestedArrayTypeName)
    ENUM_TO_STR(ContractTypeName)
    ENUM_TO_STR(AddressTypeName)
    ENUM_TO_STR(AddressPayableTypeName)
    ENUM_TO_STR(TypeConversionName)
    ENUM_TO_STR(TypeProperty)
    ENUM_TO_STR(EnumTypeName)
    ENUM_TO_STR(StructTypeName)
    ENUM_TO_STR(TupleTypeName)
    ENUM_TO_STR(MappingTypeName)
    ENUM_TO_STR(BuiltinTypeName)
    ENUM_TO_STR(ErrorTypeName)
    ENUM_TO_STR(UserDefinedTypeName)
    ENUM_TO_STR(TypeNameTError)
  default:
  {
    assert(!"Unknown type-name type");
    return "UNKNOWN";
  }
  }
}

// rule elementary-type-name
// return the type of expression
ElementaryTypeNameT get_elementary_type_name_t(const nlohmann::json &type_name)
{
  std::string typeString = type_name["typeString"].get<std::string>();
  // rule unsigned-integer-type

  if (uint_string_to_type_map.count(typeString))
  {
    return uint_string_to_type_map.at(typeString);
  }
  if (int_string_to_type_map.count(typeString))
  {
    return int_string_to_type_map.at(typeString);
  }
  if (typeString == "bool")
  {
    return BOOL;
  }
  if (typeString.find("int_const") != std::string::npos)
  {
    /**
     * For Literal, their typeString is like "int_const 100".
     * There is no additional type info (bitsize, signed/unsigned),
     * This means it will require additional type info from the parent
     * expr to create an internal type.
     */
    return INT_LITERAL;
  }
  if (typeString.find("rational_const") != std::string::npos)
  {
    // Will not be used
    return RA_LITERAL;
  }
  if (typeString.find("literal_string") == 0)
  {
    return STRING_LITERAL;
  }
  if (
    typeString == "string" || typeString == "string storage ref" ||
    typeString == "string memory" || typeString == "string storage pointer")
  {
    return STRING;
  }
  if (typeString == "address")
  {
    return ADDRESS;
  }
  if (typeString == "address payable")
  {
    return ADDRESS_PAYABLE;
  }
  if (bytesn_to_type_map.count(typeString))
  {
    // fixed-size arrays bytesN, where N is a number between 1 and 32
    return bytesn_to_type_map.at(typeString);
  }
  if (typeString.find("bytes") != std::string::npos)
  {
    // dynamic bytes array
    // e.g.
    //    bytes
    //    bytes storage ref
    //    bytes memory
    return BYTES;
  }
  log_error(
    "Got elementary-type-name typeString={}. Unsupported "
    "elementary-type-name type",
    type_name["typeString"].get<std::string>());
  abort();
}

const char *elementary_type_name_to_str(ElementaryTypeNameT type)
{
  switch (type)
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
    ENUM_TO_STR(RA_LITERAL)
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
    ENUM_TO_STR(ADDRESS)
    ENUM_TO_STR(ADDRESS_PAYABLE)
    ENUM_TO_STR(BYTES)
    ENUM_TO_STR(BYTES1)
    ENUM_TO_STR(BYTES2)
    ENUM_TO_STR(BYTES3)
    ENUM_TO_STR(BYTES4)
    ENUM_TO_STR(BYTES5)
    ENUM_TO_STR(BYTES6)
    ENUM_TO_STR(BYTES7)
    ENUM_TO_STR(BYTES8)
    ENUM_TO_STR(BYTES9)
    ENUM_TO_STR(BYTES10)
    ENUM_TO_STR(BYTES11)
    ENUM_TO_STR(BYTES12)
    ENUM_TO_STR(BYTES13)
    ENUM_TO_STR(BYTES14)
    ENUM_TO_STR(BYTES15)
    ENUM_TO_STR(BYTES16)
    ENUM_TO_STR(BYTES17)
    ENUM_TO_STR(BYTES18)
    ENUM_TO_STR(BYTES19)
    ENUM_TO_STR(BYTES20)
    ENUM_TO_STR(BYTES21)
    ENUM_TO_STR(BYTES22)
    ENUM_TO_STR(BYTES23)
    ENUM_TO_STR(BYTES24)
    ENUM_TO_STR(BYTES25)
    ENUM_TO_STR(BYTES26)
    ENUM_TO_STR(BYTES27)
    ENUM_TO_STR(BYTES28)
    ENUM_TO_STR(BYTES29)
    ENUM_TO_STR(BYTES30)
    ENUM_TO_STR(BYTES31)
    ENUM_TO_STR(BYTES32)
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

unsigned int bytesn_type_name_to_size(ElementaryTypeNameT type)
{
  return bytesn_size_map.at(type);
}

// rule parameter-list
ParameterListT get_parameter_list_t(const nlohmann::json &type_name)
{
  if (type_name["parameters"].size() == 0)
  {
    return EMPTY;
  }
  else if (type_name["parameters"].size() == 1)
  {
    return ONE_PARAM;
  }
  else if (type_name["parameters"].size() > 1)
  {
    return MORE_THAN_ONE_PARAM;
  }

  return ParameterListTError; // to make some old gcc compilers happy
}

const char *parameter_list_to_str(ParameterListT type)
{
  switch (type)
  {
    ENUM_TO_STR(EMPTY)
    ENUM_TO_STR(ONE_PARAM)
    ENUM_TO_STR(MORE_THAN_ONE_PARAM)
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
  if (
    (block["nodeType"] == "Block" || block["nodeType"] == "UncheckedBlock") &&
    block.contains("statements"))
  {
    return Statement;
  }
  else if (block["nodeType"] == "ForStatement")
  {
    return BlockForStatement;
  }
  else if (block["nodeType"] == "IfStatement")
  {
    return BlockIfStatement;
  }
  else if (block["nodeType"] == "WhileStatement")
  {
    return BlockWhileStatement;
  }
  else if (block["nodeType"] == "DoWhileStatement")
  {
    return BlockDoWhileStatement;
  }
  else if (block["nodeType"] == "ExpressionStatement")
  {
    return BlockExpressionStatement;
  }

  // fall-through
  log_error(
    "Got block nodeType={}. Unsupported block type",
    block["nodeType"].get<std::string>());
  return BlockTError;
}

const char *block_to_str(BlockT type)
{
  switch (type)
  {
    ENUM_TO_STR(Statement)
    ENUM_TO_STR(BlockForStatement)
    ENUM_TO_STR(BlockIfStatement)
    ENUM_TO_STR(BlockWhileStatement)
    ENUM_TO_STR(BlockDoWhileStatement)
    ENUM_TO_STR(BlockExpressionStatement)
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
  if (stmt["nodeType"] == "ExpressionStatement")
  {
    return ExpressionStatement;
  }
  else if (stmt["nodeType"] == "VariableDeclarationStatement")
  {
    return VariableDeclStatement;
  }
  else if (stmt["nodeType"] == "Return")
  {
    return ReturnStatement;
  }
  else if (stmt["nodeType"] == "ForStatement")
  {
    return ForStatement;
  }
  else if (stmt["nodeType"] == "Block" || stmt["nodeType"] == "UncheckedBlock")
  {
    return Block;
  }
  else if (stmt["nodeType"] == "IfStatement")
  {
    return IfStatement;
  }
  else if (stmt["nodeType"] == "WhileStatement")
  {
    return WhileStatement;
  }
  else if (stmt["nodeType"] == "DoWhileStatement")
  {
    return DoWhileStatement;
  }
  else if (stmt["nodeType"] == "Continue")
  {
    return ContinueStatement;
  }
  else if (stmt["nodeType"] == "Break")
  {
    return BreakStatement;
  }
  else if (stmt["nodeType"] == "RevertStatement")
  {
    return RevertStatement;
  }
  else if (stmt["nodeType"] == "EmitStatement")
  {
    return EmitStatement;
  }
  else if (stmt["nodeType"] == "PlaceholderStatement")
  {
    return PlaceholderStatement;
  }
  else if (stmt["nodeType"] == "TryStatement")
  {
    return TryStatement;
  }
  else if (stmt["nodeType"] == "InlineAssembly")
  {
    return InlineAssemblyStatement;
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
  switch (type)
  {
    ENUM_TO_STR(Block)
    ENUM_TO_STR(ExpressionStatement)
    ENUM_TO_STR(VariableDeclStatement)
    ENUM_TO_STR(ReturnStatement)
    ENUM_TO_STR(ForStatement)
    ENUM_TO_STR(IfStatement)
    ENUM_TO_STR(WhileStatement)
    ENUM_TO_STR(DoWhileStatement)
    ENUM_TO_STR(StatementTError)
    ENUM_TO_STR(ContinueStatement)
    ENUM_TO_STR(BreakStatement)
    ENUM_TO_STR(RevertStatement)
    ENUM_TO_STR(EmitStatement)
    ENUM_TO_STR(PlaceholderStatement)
    ENUM_TO_STR(TryStatement)
    ENUM_TO_STR(InlineAssemblyStatement)
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
  if (expr.is_null() || expr.empty())
  {
    return NullExpr;
  }
  assert(expr.contains("nodeType"));

  if (
    expr.contains("typeDescriptions") &&
    expr["typeDescriptions"].contains("typeString"))
  {
    // special handling for potential rational_const
    // e.g.
    //  - 0.5 * 1 day
    //  - 0.01 ether
    //  - uint256 x = 0.5 * 10;
    const std::string typeString = expr["typeDescriptions"]["typeString"];
    if (
      typeString.compare(0, 9, "int_const") == 0 &&
      typeString.find("...") == std::string::npos &&
      (!expr.contains("value") ||
       expr["value"].get<std::string>().find(".") != std::string::npos))
      return LiteralWithRational;
  }

  if (expr["nodeType"] == "Assignment" || expr["nodeType"] == "BinaryOperation")
  {
    return BinaryOperatorClass;
  }
  else if (expr["nodeType"] == "UnaryOperation")
  {
    return UnaryOperatorClass;
  }
  else if (expr["nodeType"] == "Conditional")
  {
    return ConditionalOperatorClass;
  }
  else if (
    expr["nodeType"] == "Identifier" && expr.contains("referencedDeclaration"))
  {
    return DeclRefExprClass;
  }
  else if (expr["nodeType"] == "Literal")
  {
    if (expr.contains("subdenomination"))
    {
      std::string unit = expr["subdenomination"];

      if (unit == "wei")
        return LiteralWithWei;
      else if (unit == "gwei")
        return LiteralWithGwei;
      else if (unit == "szabo")
        return LiteralWithSzabo;
      else if (unit == "finney")
        return LiteralWithFinney;
      else if (unit == "ether")
        return LiteralWithEther;
      else if (unit == "seconds")
        return LiteralWithSeconds;
      else if (unit == "minutes")
        return LiteralWithMinutes;
      else if (unit == "hours")
        return LiteralWithHours;
      else if (unit == "days")
        return LiteralWithDays;
      else if (unit == "weeks")
        return LiteralWithWeeks;
      else if (unit == "years")
        return LiteralWithYears;
      else
        return LiteralWithUnknownUnit;
    }

    return Literal;
  }
  else if (expr["nodeType"] == "TupleExpression")
  {
    return Tuple;
  }
  else if (
    expr["nodeType"] == "Mapping" ||
    (expr["nodeType"] == "VariableDeclaration" &&
     expr["typeName"]["nodeType"] == "Mapping"))
  {
    return Mapping;
  }
  else if (expr["nodeType"] == "FunctionCallOptions")
  {
    // if (expr["expression"]["nodeType"] == "NewExpression")
    //   return NewExpression;
    return CallOptionsExprClass;
  }
  else if (expr["nodeType"] == "FunctionCall")
  {
    if (
      expr["expression"]["nodeType"] == "NewExpression" ||
      (expr["expression"]["nodeType"] == "FunctionCallOptions" &&
       expr["expression"]["expression"]["nodeType"] == "NewExpression"))
      return NewExpression;
    if (expr["kind"] == "typeConversion")
      return TypeConversionExpression;
    if (
      get_type_name_t(expr["typeDescriptions"]) ==
      SolidityGrammar::TypeProperty)
      return TypePropertyExpression;

    return CallExprClass;
  }
  else if (expr["nodeType"] == "MemberAccess")
  {
    assert(expr.contains("expression"));
    SolidityGrammar::TypeNameT type_name =
      get_type_name_t(expr["expression"]["typeDescriptions"]);
    if (
      expr.contains("referencedDeclaration") &&
      is_sol_library_function(expr["referencedDeclaration"].get<int>()))
      return LibraryMemberCall;
    else if (type_name == SolidityGrammar::TypeNameT::StructTypeName)
      return StructMemberCall;
    else if (type_name == SolidityGrammar::TypeNameT::EnumTypeName)
      return EnumMemberCall;
    else if (type_name == SolidityGrammar::TypeNameT::ContractTypeName)
      return ContractMemberCall;
    else if (is_address_member_call(expr))
      return AddressMemberCall;
    else if (type_name == SolidityGrammar::TypeNameT::TypeConversionName)
      return TypeMemberCall;
    else
      //TODO Assume it's a builtin member
      // due to that the BuiltinTypeName cannot cover all the builtin member
      // e.g. string.concat ==> TypeConversionName
      return BuiltinMemberCall;
  }
  else if (expr["nodeType"] == "ImplicitCastExprClass")
  {
    return ImplicitCastExprClass;
  }
  else if (expr["nodeType"] == "IndexAccess")
  {
    return IndexAccess;
  }
  else if (expr["nodeType"] == "IndexRangeAccess")
  {
    return IndexRangeAccess;
  }

  // fall-through
  log_error(
    "Got expression nodeType={}. Unsupported expression type",
    expr["nodeType"].get<std::string>());
  return ExpressionTError;
}

ExpressionT get_unary_expr_operator_t(const nlohmann::json &expr, bool uo_pre)
{
  if (expr["operator"] == "--")
  {
    if (uo_pre)
      return UO_PreDec;
    else
      return UO_PostDec;
  }
  else if (expr["operator"] == "++")
  {
    if (uo_pre)
      return UO_PreInc;
    else
      return UO_PostInc;
  }
  else if (expr["operator"] == "-")
  {
    return UO_Minus;
  }
  else if (expr["operator"] == "~")
  {
    return UO_Not;
  }
  else if (expr["operator"] == "!")
  {
    return UO_LNot;
  }
  else if (expr["operator"] == "delete")
  {
    return UO_Delete;
  }
  else
  {
    log_error(
      "Got expression operator={}. Unsupported expression operator",
      expr["operator"].get<std::string>());

    abort();
  }
}

ExpressionT get_expr_operator_t(const nlohmann::json &expr)
{
  if (expr["operator"] == "=")
  {
    return BO_Assign;
  }
  else if (expr["operator"] == "+")
  {
    return BO_Add;
  }
  else if (expr["operator"] == "-")
  {
    return BO_Sub;
  }
  else if (expr["operator"] == "*")
  {
    return BO_Mul;
  }
  else if (expr["operator"] == "/")
  {
    return BO_Div;
  }
  else if (expr["operator"] == "%")
  {
    return BO_Rem;
  }
  else if (expr["operator"] == "<<")
  {
    return BO_Shl;
  }
  else if (expr["operator"] == ">>")
  {
    return BO_Shr;
  }
  else if (expr["operator"] == "&")
  {
    return BO_And;
  }
  else if (expr["operator"] == "^")
  {
    return BO_Xor;
  }
  else if (expr["operator"] == "|")
  {
    return BO_Or;
  }
  else if (expr["operator"] == ">")
  {
    return BO_GT;
  }
  else if (expr["operator"] == "<")
  {
    return BO_LT;
  }
  else if (expr["operator"] == ">=")
  {
    return BO_GE;
  }
  else if (expr["operator"] == "<=")
  {
    return BO_LE;
  }
  else if (expr["operator"] == "!=")
  {
    return BO_NE;
  }
  else if (expr["operator"] == "==")
  {
    return BO_EQ;
  }
  else if (expr["operator"] == "&&")
  {
    return BO_LAnd;
  }
  else if (expr["operator"] == "||")
  {
    return BO_LOr;
  }
  else if (expr["operator"] == "+=")
  {
    return BO_AddAssign;
  }
  else if (expr["operator"] == "-=")
  {
    return BO_SubAssign;
  }
  else if (expr["operator"] == "*=")
  {
    return BO_MulAssign;
  }
  else if (expr["operator"] == "/=")
  {
    return BO_DivAssign;
  }
  else if (expr["operator"] == "%=")
  {
    return BO_RemAssign;
  }
  else if (expr["operator"] == "<<=")
  {
    return BO_ShlAssign;
  }
  else if (expr["operator"] == ">>=")
  {
    return BO_ShrAssign;
  }
  else if (expr["operator"] == "&=")
  {
    return BO_AndAssign;
  }
  else if (expr["operator"] == "^=")
  {
    return BO_XorAssign;
  }
  else if (expr["operator"] == "|=")
  {
    return BO_OrAssign;
  }
  else if (expr["operator"] == "**")
  {
    return BO_Pow;
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
  switch (type)
  {
    ENUM_TO_STR(BinaryOperatorClass)
    ENUM_TO_STR(BO_Assign)
    ENUM_TO_STR(BO_Add)
    ENUM_TO_STR(BO_Sub)
    ENUM_TO_STR(BO_Mul)
    ENUM_TO_STR(BO_Div)
    ENUM_TO_STR(BO_Rem)

    ENUM_TO_STR(BO_Shl)
    ENUM_TO_STR(BO_Shr)
    ENUM_TO_STR(BO_And)
    ENUM_TO_STR(BO_Xor)
    ENUM_TO_STR(BO_Or)

    ENUM_TO_STR(BO_GT)
    ENUM_TO_STR(BO_LT)
    ENUM_TO_STR(BO_GE)
    ENUM_TO_STR(BO_LE)
    ENUM_TO_STR(BO_NE)
    ENUM_TO_STR(BO_EQ)
    ENUM_TO_STR(BO_LAnd)
    ENUM_TO_STR(BO_LOr)

    ENUM_TO_STR(BO_AddAssign)
    ENUM_TO_STR(BO_SubAssign)
    ENUM_TO_STR(BO_MulAssign)
    ENUM_TO_STR(BO_DivAssign)
    ENUM_TO_STR(BO_RemAssign)
    ENUM_TO_STR(BO_ShlAssign)
    ENUM_TO_STR(BO_ShrAssign)
    ENUM_TO_STR(BO_AndAssign)
    ENUM_TO_STR(BO_XorAssign)
    ENUM_TO_STR(BO_OrAssign)
    ENUM_TO_STR(BO_Pow)

    ENUM_TO_STR(UnaryOperatorClass)
    ENUM_TO_STR(UO_PreDec)
    ENUM_TO_STR(UO_PreInc)
    ENUM_TO_STR(UO_PostDec)
    ENUM_TO_STR(UO_PostInc)
    ENUM_TO_STR(UO_Minus)
    ENUM_TO_STR(UO_Not)
    ENUM_TO_STR(UO_LNot)
    ENUM_TO_STR(UO_Delete)

    ENUM_TO_STR(ConditionalOperatorClass)

    ENUM_TO_STR(DeclRefExprClass)
    ENUM_TO_STR(Literal)
    ENUM_TO_STR(LiteralWithRational)
    ENUM_TO_STR(LiteralWithWei)
    ENUM_TO_STR(LiteralWithGwei)
    ENUM_TO_STR(LiteralWithSzabo)
    ENUM_TO_STR(LiteralWithFinney)
    ENUM_TO_STR(LiteralWithEther)
    ENUM_TO_STR(LiteralWithSeconds)
    ENUM_TO_STR(LiteralWithMinutes)
    ENUM_TO_STR(LiteralWithHours)
    ENUM_TO_STR(LiteralWithDays)
    ENUM_TO_STR(LiteralWithWeeks)
    ENUM_TO_STR(LiteralWithYears)
    ENUM_TO_STR(LiteralWithUnknownUnit)
    ENUM_TO_STR(Tuple)
    ENUM_TO_STR(Mapping)
    ENUM_TO_STR(CallExprClass)
    ENUM_TO_STR(CallOptionsExprClass)
    ENUM_TO_STR(ImplicitCastExprClass)
    ENUM_TO_STR(IndexAccess)
    ENUM_TO_STR(IndexRangeAccess)
    ENUM_TO_STR(NewExpression)
    ENUM_TO_STR(AddressMemberCall)
    ENUM_TO_STR(LibraryMemberCall)
    ENUM_TO_STR(ContractMemberCall)
    ENUM_TO_STR(StructMemberCall)
    ENUM_TO_STR(EnumMemberCall)
    ENUM_TO_STR(BuiltinMemberCall)
    ENUM_TO_STR(TypeMemberCall)
    ENUM_TO_STR(TypeConversionExpression)
    ENUM_TO_STR(TypePropertyExpression)
    ENUM_TO_STR(NullExpr)
    ENUM_TO_STR(ExpressionTError)
  default:
  {
    assert(!"Unknown expression type");
    return "UNKNOWN";
  }
  }
}

// auxiliary type to convert function call
FunctionDeclRefT get_func_decl_ref_t(const nlohmann::json &decl)
{
  assert(
    decl["nodeType"] == "FunctionDefinition" ||
    decl["nodeType"] == "EventDefinition");
  if (
    decl["parameters"]["parameters"].size() == 0 ||
    (decl.contains("kind") && decl["kind"] == "constructor"))
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
  switch (type)
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
  if (cast == "LValueToRValue")
  {
    return LValueToRValue;
  }
  else if (cast == "FunctionToPointerDecay")
  {
    return FunctionToPointerDecay;
  }
  else if (cast == "ArrayToPointerDecay")
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
  switch (type)
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

VisibilityT get_access_t(const nlohmann::json &ast_node)
{
  if (!ast_node.contains("visibility"))
    return UnknownT;
  std::string access = ast_node["visibility"].get<std::string>();
  if (access == "public")
  {
    return PublicT;
  }
  else if (access == "private")
  {
    return PrivateT;
  }
  else if (access == "external")
  {
    return ExternalT;
  }
  else if (access == "internal")
  {
    return InternalT;
  }
  else
  {
    log_error("Unknown Visibility");
    abort();
  }
  return UnknownT;
}

// ====== SolType enum support ======

#define SOL_ENUM_TO_STR(s)                                                     \
  case SolType::s:                                                             \
    return #s;

const char *sol_type_to_str(SolType t)
{
  switch (t)
  {
    SOL_ENUM_TO_STR(UINT8)
    SOL_ENUM_TO_STR(UINT16)
    SOL_ENUM_TO_STR(UINT24)
    SOL_ENUM_TO_STR(UINT32)
    SOL_ENUM_TO_STR(UINT40)
    SOL_ENUM_TO_STR(UINT48)
    SOL_ENUM_TO_STR(UINT56)
    SOL_ENUM_TO_STR(UINT64)
    SOL_ENUM_TO_STR(UINT72)
    SOL_ENUM_TO_STR(UINT80)
    SOL_ENUM_TO_STR(UINT88)
    SOL_ENUM_TO_STR(UINT96)
    SOL_ENUM_TO_STR(UINT104)
    SOL_ENUM_TO_STR(UINT112)
    SOL_ENUM_TO_STR(UINT120)
    SOL_ENUM_TO_STR(UINT128)
    SOL_ENUM_TO_STR(UINT136)
    SOL_ENUM_TO_STR(UINT144)
    SOL_ENUM_TO_STR(UINT152)
    SOL_ENUM_TO_STR(UINT160)
    SOL_ENUM_TO_STR(UINT168)
    SOL_ENUM_TO_STR(UINT176)
    SOL_ENUM_TO_STR(UINT184)
    SOL_ENUM_TO_STR(UINT192)
    SOL_ENUM_TO_STR(UINT200)
    SOL_ENUM_TO_STR(UINT208)
    SOL_ENUM_TO_STR(UINT216)
    SOL_ENUM_TO_STR(UINT224)
    SOL_ENUM_TO_STR(UINT232)
    SOL_ENUM_TO_STR(UINT240)
    SOL_ENUM_TO_STR(UINT248)
    SOL_ENUM_TO_STR(UINT256)
    SOL_ENUM_TO_STR(INT8)
    SOL_ENUM_TO_STR(INT16)
    SOL_ENUM_TO_STR(INT24)
    SOL_ENUM_TO_STR(INT32)
    SOL_ENUM_TO_STR(INT40)
    SOL_ENUM_TO_STR(INT48)
    SOL_ENUM_TO_STR(INT56)
    SOL_ENUM_TO_STR(INT64)
    SOL_ENUM_TO_STR(INT72)
    SOL_ENUM_TO_STR(INT80)
    SOL_ENUM_TO_STR(INT88)
    SOL_ENUM_TO_STR(INT96)
    SOL_ENUM_TO_STR(INT104)
    SOL_ENUM_TO_STR(INT112)
    SOL_ENUM_TO_STR(INT120)
    SOL_ENUM_TO_STR(INT128)
    SOL_ENUM_TO_STR(INT136)
    SOL_ENUM_TO_STR(INT144)
    SOL_ENUM_TO_STR(INT152)
    SOL_ENUM_TO_STR(INT160)
    SOL_ENUM_TO_STR(INT168)
    SOL_ENUM_TO_STR(INT176)
    SOL_ENUM_TO_STR(INT184)
    SOL_ENUM_TO_STR(INT192)
    SOL_ENUM_TO_STR(INT200)
    SOL_ENUM_TO_STR(INT208)
    SOL_ENUM_TO_STR(INT216)
    SOL_ENUM_TO_STR(INT224)
    SOL_ENUM_TO_STR(INT232)
    SOL_ENUM_TO_STR(INT240)
    SOL_ENUM_TO_STR(INT248)
    SOL_ENUM_TO_STR(INT256)
    SOL_ENUM_TO_STR(BOOL)
    SOL_ENUM_TO_STR(ADDRESS)
    SOL_ENUM_TO_STR(ADDRESS_PAYABLE)
    SOL_ENUM_TO_STR(STRING)
    SOL_ENUM_TO_STR(ENUM)
    SOL_ENUM_TO_STR(BYTES1)
    SOL_ENUM_TO_STR(BYTES2)
    SOL_ENUM_TO_STR(BYTES3)
    SOL_ENUM_TO_STR(BYTES4)
    SOL_ENUM_TO_STR(BYTES5)
    SOL_ENUM_TO_STR(BYTES6)
    SOL_ENUM_TO_STR(BYTES7)
    SOL_ENUM_TO_STR(BYTES8)
    SOL_ENUM_TO_STR(BYTES9)
    SOL_ENUM_TO_STR(BYTES10)
    SOL_ENUM_TO_STR(BYTES11)
    SOL_ENUM_TO_STR(BYTES12)
    SOL_ENUM_TO_STR(BYTES13)
    SOL_ENUM_TO_STR(BYTES14)
    SOL_ENUM_TO_STR(BYTES15)
    SOL_ENUM_TO_STR(BYTES16)
    SOL_ENUM_TO_STR(BYTES17)
    SOL_ENUM_TO_STR(BYTES18)
    SOL_ENUM_TO_STR(BYTES19)
    SOL_ENUM_TO_STR(BYTES20)
    SOL_ENUM_TO_STR(BYTES21)
    SOL_ENUM_TO_STR(BYTES22)
    SOL_ENUM_TO_STR(BYTES23)
    SOL_ENUM_TO_STR(BYTES24)
    SOL_ENUM_TO_STR(BYTES25)
    SOL_ENUM_TO_STR(BYTES26)
    SOL_ENUM_TO_STR(BYTES27)
    SOL_ENUM_TO_STR(BYTES28)
    SOL_ENUM_TO_STR(BYTES29)
    SOL_ENUM_TO_STR(BYTES30)
    SOL_ENUM_TO_STR(BYTES31)
    SOL_ENUM_TO_STR(BYTES32)
  case SolType::BYTES_DYN:
    return "BytesDynamic";
  case SolType::BYTES_STATIC:
    return "BytesStatic";
    SOL_ENUM_TO_STR(INT_CONST)
    SOL_ENUM_TO_STR(STRING_LITERAL)
    SOL_ENUM_TO_STR(ARRAY)
    SOL_ENUM_TO_STR(ARRAY_LITERAL)
    SOL_ENUM_TO_STR(DYNARRAY)
    SOL_ENUM_TO_STR(ARRAY_CALLOC)
    SOL_ENUM_TO_STR(MAPPING)
    SOL_ENUM_TO_STR(STRUCT)
    SOL_ENUM_TO_STR(CONTRACT)
    SOL_ENUM_TO_STR(LIBRARY)
    SOL_ENUM_TO_STR(TUPLE_RETURNS)
    SOL_ENUM_TO_STR(TUPLE_INSTANCE)
    SOL_ENUM_TO_STR(UNSET)
  }
  return "UNSET";
}

#undef SOL_ENUM_TO_STR

SolType str_to_sol_type(const std::string &s)
{
  static const std::unordered_map<std::string, SolType> map = {
    {"UINT8", SolType::UINT8},
    {"UINT16", SolType::UINT16},
    {"UINT24", SolType::UINT24},
    {"UINT32", SolType::UINT32},
    {"UINT40", SolType::UINT40},
    {"UINT48", SolType::UINT48},
    {"UINT56", SolType::UINT56},
    {"UINT64", SolType::UINT64},
    {"UINT72", SolType::UINT72},
    {"UINT80", SolType::UINT80},
    {"UINT88", SolType::UINT88},
    {"UINT96", SolType::UINT96},
    {"UINT104", SolType::UINT104},
    {"UINT112", SolType::UINT112},
    {"UINT120", SolType::UINT120},
    {"UINT128", SolType::UINT128},
    {"UINT136", SolType::UINT136},
    {"UINT144", SolType::UINT144},
    {"UINT152", SolType::UINT152},
    {"UINT160", SolType::UINT160},
    {"UINT168", SolType::UINT168},
    {"UINT176", SolType::UINT176},
    {"UINT184", SolType::UINT184},
    {"UINT192", SolType::UINT192},
    {"UINT200", SolType::UINT200},
    {"UINT208", SolType::UINT208},
    {"UINT216", SolType::UINT216},
    {"UINT224", SolType::UINT224},
    {"UINT232", SolType::UINT232},
    {"UINT240", SolType::UINT240},
    {"UINT248", SolType::UINT248},
    {"UINT256", SolType::UINT256},
    {"INT8", SolType::INT8},
    {"INT16", SolType::INT16},
    {"INT24", SolType::INT24},
    {"INT32", SolType::INT32},
    {"INT40", SolType::INT40},
    {"INT48", SolType::INT48},
    {"INT56", SolType::INT56},
    {"INT64", SolType::INT64},
    {"INT72", SolType::INT72},
    {"INT80", SolType::INT80},
    {"INT88", SolType::INT88},
    {"INT96", SolType::INT96},
    {"INT104", SolType::INT104},
    {"INT112", SolType::INT112},
    {"INT120", SolType::INT120},
    {"INT128", SolType::INT128},
    {"INT136", SolType::INT136},
    {"INT144", SolType::INT144},
    {"INT152", SolType::INT152},
    {"INT160", SolType::INT160},
    {"INT168", SolType::INT168},
    {"INT176", SolType::INT176},
    {"INT184", SolType::INT184},
    {"INT192", SolType::INT192},
    {"INT200", SolType::INT200},
    {"INT208", SolType::INT208},
    {"INT216", SolType::INT216},
    {"INT224", SolType::INT224},
    {"INT232", SolType::INT232},
    {"INT240", SolType::INT240},
    {"INT248", SolType::INT248},
    {"INT256", SolType::INT256},
    {"BOOL", SolType::BOOL},
    {"ADDRESS", SolType::ADDRESS},
    {"ADDRESS_PAYABLE", SolType::ADDRESS_PAYABLE},
    {"STRING", SolType::STRING},
    {"ENUM", SolType::ENUM},
    {"BYTES1", SolType::BYTES1},
    {"BYTES2", SolType::BYTES2},
    {"BYTES3", SolType::BYTES3},
    {"BYTES4", SolType::BYTES4},
    {"BYTES5", SolType::BYTES5},
    {"BYTES6", SolType::BYTES6},
    {"BYTES7", SolType::BYTES7},
    {"BYTES8", SolType::BYTES8},
    {"BYTES9", SolType::BYTES9},
    {"BYTES10", SolType::BYTES10},
    {"BYTES11", SolType::BYTES11},
    {"BYTES12", SolType::BYTES12},
    {"BYTES13", SolType::BYTES13},
    {"BYTES14", SolType::BYTES14},
    {"BYTES15", SolType::BYTES15},
    {"BYTES16", SolType::BYTES16},
    {"BYTES17", SolType::BYTES17},
    {"BYTES18", SolType::BYTES18},
    {"BYTES19", SolType::BYTES19},
    {"BYTES20", SolType::BYTES20},
    {"BYTES21", SolType::BYTES21},
    {"BYTES22", SolType::BYTES22},
    {"BYTES23", SolType::BYTES23},
    {"BYTES24", SolType::BYTES24},
    {"BYTES25", SolType::BYTES25},
    {"BYTES26", SolType::BYTES26},
    {"BYTES27", SolType::BYTES27},
    {"BYTES28", SolType::BYTES28},
    {"BYTES29", SolType::BYTES29},
    {"BYTES30", SolType::BYTES30},
    {"BYTES31", SolType::BYTES31},
    {"BYTES32", SolType::BYTES32},
    {"BytesDynamic", SolType::BYTES_DYN},
    {"BytesStatic", SolType::BYTES_STATIC},
    {"INT_CONST", SolType::INT_CONST},
    {"STRING_LITERAL", SolType::STRING_LITERAL},
    {"ARRAY", SolType::ARRAY},
    {"ARRAY_LITERAL", SolType::ARRAY_LITERAL},
    {"DYNARRAY", SolType::DYNARRAY},
    {"ARRAY_CALLOC", SolType::ARRAY_CALLOC},
    {"MAPPING", SolType::MAPPING},
    {"STRUCT", SolType::STRUCT},
    {"CONTRACT", SolType::CONTRACT},
    {"LIBRARY", SolType::LIBRARY},
    {"TUPLE_RETURNS", SolType::TUPLE_RETURNS},
    {"TUPLE_INSTANCE", SolType::TUPLE_INSTANCE},
  };
  auto it = map.find(s);
  return it != map.end() ? it->second : SolType::UNSET;
}

SolType elementary_to_sol_type(ElementaryTypeNameT t)
{
  switch (t)
  {
  case UINT8:
    return SolType::UINT8;
  case UINT16:
    return SolType::UINT16;
  case UINT24:
    return SolType::UINT24;
  case UINT32:
    return SolType::UINT32;
  case UINT40:
    return SolType::UINT40;
  case UINT48:
    return SolType::UINT48;
  case UINT56:
    return SolType::UINT56;
  case UINT64:
    return SolType::UINT64;
  case UINT72:
    return SolType::UINT72;
  case UINT80:
    return SolType::UINT80;
  case UINT88:
    return SolType::UINT88;
  case UINT96:
    return SolType::UINT96;
  case UINT104:
    return SolType::UINT104;
  case UINT112:
    return SolType::UINT112;
  case UINT120:
    return SolType::UINT120;
  case UINT128:
    return SolType::UINT128;
  case UINT136:
    return SolType::UINT136;
  case UINT144:
    return SolType::UINT144;
  case UINT152:
    return SolType::UINT152;
  case UINT160:
    return SolType::UINT160;
  case UINT168:
    return SolType::UINT168;
  case UINT176:
    return SolType::UINT176;
  case UINT184:
    return SolType::UINT184;
  case UINT192:
    return SolType::UINT192;
  case UINT200:
    return SolType::UINT200;
  case UINT208:
    return SolType::UINT208;
  case UINT216:
    return SolType::UINT216;
  case UINT224:
    return SolType::UINT224;
  case UINT232:
    return SolType::UINT232;
  case UINT240:
    return SolType::UINT240;
  case UINT248:
    return SolType::UINT248;
  case UINT256:
    return SolType::UINT256;
  case INT8:
    return SolType::INT8;
  case INT16:
    return SolType::INT16;
  case INT24:
    return SolType::INT24;
  case INT32:
    return SolType::INT32;
  case INT40:
    return SolType::INT40;
  case INT48:
    return SolType::INT48;
  case INT56:
    return SolType::INT56;
  case INT64:
    return SolType::INT64;
  case INT72:
    return SolType::INT72;
  case INT80:
    return SolType::INT80;
  case INT88:
    return SolType::INT88;
  case INT96:
    return SolType::INT96;
  case INT104:
    return SolType::INT104;
  case INT112:
    return SolType::INT112;
  case INT120:
    return SolType::INT120;
  case INT128:
    return SolType::INT128;
  case INT136:
    return SolType::INT136;
  case INT144:
    return SolType::INT144;
  case INT152:
    return SolType::INT152;
  case INT160:
    return SolType::INT160;
  case INT168:
    return SolType::INT168;
  case INT176:
    return SolType::INT176;
  case INT184:
    return SolType::INT184;
  case INT192:
    return SolType::INT192;
  case INT200:
    return SolType::INT200;
  case INT208:
    return SolType::INT208;
  case INT216:
    return SolType::INT216;
  case INT224:
    return SolType::INT224;
  case INT232:
    return SolType::INT232;
  case INT240:
    return SolType::INT240;
  case INT248:
    return SolType::INT248;
  case INT256:
    return SolType::INT256;
  case BOOL:
    return SolType::BOOL;
  case ADDRESS:
    return SolType::ADDRESS;
  case ADDRESS_PAYABLE:
    return SolType::ADDRESS_PAYABLE;
  case STRING:
    return SolType::STRING;
  case STRING_LITERAL:
    return SolType::STRING_LITERAL;
  case BYTES:
    return SolType::BYTES_DYN;
  case BYTES1:
    return SolType::BYTES1;
  case BYTES2:
    return SolType::BYTES2;
  case BYTES3:
    return SolType::BYTES3;
  case BYTES4:
    return SolType::BYTES4;
  case BYTES5:
    return SolType::BYTES5;
  case BYTES6:
    return SolType::BYTES6;
  case BYTES7:
    return SolType::BYTES7;
  case BYTES8:
    return SolType::BYTES8;
  case BYTES9:
    return SolType::BYTES9;
  case BYTES10:
    return SolType::BYTES10;
  case BYTES11:
    return SolType::BYTES11;
  case BYTES12:
    return SolType::BYTES12;
  case BYTES13:
    return SolType::BYTES13;
  case BYTES14:
    return SolType::BYTES14;
  case BYTES15:
    return SolType::BYTES15;
  case BYTES16:
    return SolType::BYTES16;
  case BYTES17:
    return SolType::BYTES17;
  case BYTES18:
    return SolType::BYTES18;
  case BYTES19:
    return SolType::BYTES19;
  case BYTES20:
    return SolType::BYTES20;
  case BYTES21:
    return SolType::BYTES21;
  case BYTES22:
    return SolType::BYTES22;
  case BYTES23:
    return SolType::BYTES23;
  case BYTES24:
    return SolType::BYTES24;
  case BYTES25:
    return SolType::BYTES25;
  case BYTES26:
    return SolType::BYTES26;
  case BYTES27:
    return SolType::BYTES27;
  case BYTES28:
    return SolType::BYTES28;
  case BYTES29:
    return SolType::BYTES29;
  case BYTES30:
    return SolType::BYTES30;
  case BYTES31:
    return SolType::BYTES31;
  case BYTES32:
    return SolType::BYTES32;
  default:
    return SolType::UNSET;
  }
}

bool is_uint_type(SolType t)
{
  return t >= SolType::UINT8 && t <= SolType::UINT256;
}

bool is_int_type(SolType t)
{
  return t >= SolType::INT8 && t <= SolType::INT256;
}

bool is_integer_type(SolType t)
{
  return is_uint_type(t) || is_int_type(t);
}

bool is_bytesN_type(SolType t)
{
  return t >= SolType::BYTES1 && t <= SolType::BYTES32;
}

bool is_bytes_type(SolType t)
{
  return is_bytesN_type(t) || t == SolType::BYTES_DYN ||
         t == SolType::BYTES_STATIC;
}

bool is_address_type(SolType t)
{
  return t == SolType::ADDRESS || t == SolType::ADDRESS_PAYABLE;
}

}; // namespace SolidityGrammar
