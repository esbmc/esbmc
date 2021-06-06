#include <solidity-ast-frontend/solidity_type.h>

namespace SolidityTypes
{
  declClass get_decl_class(const std::string& kind)
  {
    if (kind == "DeclLabel")
    {
      return DeclLabel;
    }
    else if (kind == "DeclFunction")
    {
      return DeclFunction;
    }
    else
    {
      assert(!"Unsupported declClass");
    }
    return DeclError;
  }

  const char* declClass_to_str(declClass the_decl)
  {
    switch(the_decl)
    {
      ENUM_TO_STR(DeclLabel)
      ENUM_TO_STR(DeclFunction)
      ENUM_TO_STR(DeclError)
      default:
      {
        assert(!"Unknown declClass");
        return "UNKNOWN";
      }
    }
  }

  typeClass get_type_class(const std::string& kind)
  {
    if (kind == "TypeBuiltin")
    {
      return TypeBuiltin;
    }
    else
    {
      assert(!"Unsupported declClass");
    }
    return TypeError;
  }

  const char* typeClass_to_str(typeClass the_type)
  {
    switch(the_type)
    {
      ENUM_TO_STR(TypeBuiltin)
      ENUM_TO_STR(TypeError)
      default:
      {
        assert(!"Unknown typeClass");
        return "UNKNOWN";
      }
    }
  }
};
