#include <solidity-frontend/solidity_grammar.h>

namespace SolidityGrammar
{
  ContractBodyElementT get_contract_body_element_t(const nlohmann::json element)
  {
    if (element["nodeType"] == "VariableDeclaration" &&
        element["stateVariable"] == true)
    {
      return StateVarDecl;
    }
    else
    {
      assert(!"Unsupported contract-body-element type");
    }
    return ContractBodyElementTError;
  }

  const char* contract_body_element_to_str(ContractBodyElementT type)
  {
    switch(type)
    {
      ENUM_TO_STR(StateVarDecl)
      ENUM_TO_STR(ContractBodyElementTError)
      default:
      {
        assert(!"Unknown contract-body-element type");
        return "UNKNOWN";
      }
    }
  }

  TypeNameT get_type_name_t(const nlohmann::json type_name)
  {
    if (type_name["nodeType"] == "ElementaryTypeName")
    {
      return ElementaryTypeName;
    }
    else
    {
      assert(!"Unsupported type-name type");
    }
    return TypeNameTError;
  }

  const char* type_name_to_str(TypeNameT type)
  {
    switch(type)
    {
      ENUM_TO_STR(ElementaryTypeName)
      ENUM_TO_STR(TypeNameTError)
      default:
      {
        assert(!"Unknown type-name type");
        return "UNKNOWN";
      }
    }
  }

  ElementaryTypeNameT get_elementary_type_name_t(const nlohmann::json type_name)
  {
    // rule unsigned-integer-type
    if (type_name["name"] == "uint8")
    {
      return UINT8;
    }
    else
    {
      assert(!"Unsupported elementary-type-name type");
    }
    return ElementaryTypeNameTError;
  }

  const char* elementary_type_name_to_str(ElementaryTypeNameT type)
  {
    switch(type)
    {
      ENUM_TO_STR(UINT8)
      ENUM_TO_STR(ElementaryTypeNameTError)
      default:
      {
        assert(!"Unknown elementary-type-name type");
        return "UNKNOWN";
      }
    }
  }
};
