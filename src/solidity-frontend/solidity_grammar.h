#ifndef SOLIDITY_GRAMMAR_H_
#define SOLIDITY_GRAMMAR_H_

#include <map>
#include <string>
#include <nlohmann/json.hpp>

#define ENUM_TO_STR(s) case s: { return #s; }

namespace SolidityGrammar
{
  // rule contract-body-element
  enum ContractBodyElementT
  {
    StateVarDecl = 0, // rule state-variable-declaration
    FunctionDef,      // rule function-definition
    ContractBodyElementTError
  };
  ContractBodyElementT get_contract_body_element_t(const nlohmann::json &element);
  const char* contract_body_element_to_str(ContractBodyElementT type);

  // rule type-name
  enum TypeNameT
  {
    ElementaryTypeName = 0, // rule elementary-type-name
    ParameterList,          // rule parameter-list. Strictly, this should not be here. Just a workaround
    TypeNameTError
  };
  TypeNameT get_type_name_t(const nlohmann::json &type_name);
  const char* type_name_to_str(TypeNameT type);

  // rule elementary-type-name
  enum ElementaryTypeNameT
  {
    // rule unsigned-integer-type
    UINT8 = 0,

    // TODO: rule address
    // TODO: rule address payable
    // TODO: rule bool
    // TODO: rule string
    // TODO: rule bytes
    // TODO: rule signed-integer-type
    // TODO: rule e
    // TODO: fixed-bytes
    // TODO: fixed
    // TODO: ufixed
    ElementaryTypeNameTError
  };
  ElementaryTypeNameT get_elementary_type_name_t(const nlohmann::json &type_name);
  const char* elementary_type_name_to_str(ElementaryTypeNameT type);

  // rule parameter-list
  enum ParameterListT
  {
    EMPTY = 0, // In Solidity, "void" means an empty parameter list
    NONEMPTY,
    ParameterListTError
  };
  ParameterListT get_parameter_list_t(const nlohmann::json &type_name);
  const char* parameter_list_to_str(ParameterListT type);

  // rule block
  enum BlockT
  {
    Statement = 0,
    UncheckedBlock,
    BlockTError
  };
  BlockT get_block_t(const nlohmann::json &block);
  const char* block_to_str(BlockT type);
}; // end of SolidityGrammar

#endif /* SOLIDITY_GRAMMAR_H_ */
