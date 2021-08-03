#include <solidity-frontend/solidity_grammar.h>

namespace SolidityGrammar
{
  // rule contract-body-element
  ContractBodyElementT get_contract_body_element_t(const nlohmann::json &element)
  {
    if (element["nodeType"] == "VariableDeclaration" &&
        element["stateVariable"] == true)
    {
      return StateVarDecl;
    }
    else if (element["nodeType"] == "FunctionDefinition" &&
             element["kind"] == "function")
    {
      return FunctionDef;
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
    if (type_name["nodeType"] == "ElementaryTypeName")
    {
      return ElementaryTypeName;
    }
    else if (type_name["nodeType"] == "ParameterList")
    {
      return ParameterList;
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

  // rule elementary-type-name
  ElementaryTypeNameT get_elementary_type_name_t(const nlohmann::json &type_name)
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

  // rule parameter-list
  ParameterListT get_parameter_list_t(const nlohmann::json &type_name)
  {
    if (type_name["parameters"].size() == 0)
    {
      return EMPTY;
    }
    else
    {
      return NONEMPTY;
    }

    return ParameterListTError; // to make some old gcc compilers happy
  }

  const char* parameter_list_to_str(ParameterListT type)
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
    if (block["nodeType"] == "Block" && block.contains("statements"))
    {
      return Statement;
    }
    else
    {
      assert(!"Unsupported block type");
    }
    return BlockTError;
  }

  const char* block_to_str(BlockT type)
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
    if (stmt["nodeType"] == "ExpressionStatement")
    {
      return ExpressionStatement;
    }
    else
    {
      assert(!"Unsupported statement type");
    }
    return StatementTError;
  }

  const char* statement_to_str(StatementT type)
  {
    switch(type)
    {
      ENUM_TO_STR(Block)
      ENUM_TO_STR(ExpressionStatement)
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
    if (expr["nodeType"] == "Assignment")
    {
      return BinaryOperator;
    }
    else
    {
      assert(!"Unsupported expression type");
    }
    return ExpressionTError;
  }

  const char* expression_to_str(ExpressionT type)
  {
    switch(type)
    {
      ENUM_TO_STR(BinaryOperator)
      ENUM_TO_STR(ExpressionTError)
      default:
      {
        assert(!"Unknown expression type");
        return "UNKNOWN";
      }
    }
  }
};
