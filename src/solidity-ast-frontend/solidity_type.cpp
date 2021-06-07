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
    else if (kind == "DeclParmVar")
    {
      return DeclParmVar;
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
      ENUM_TO_STR(DeclParmVar)
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
      assert(!"Unsupported typeClass");
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

  builInTypes get_builtin_type(const std::string& kind)
  {
    if (kind == "BuiltInVoid")
    {
      return BuiltInVoid;
    }
    else if (kind == "BuiltInBool")
    {
      return BuiltInBool;
    }
    else
    {
      assert(!"Unsupported builtInTypes");
    }
    return BuiltInError;
  }
  const char* builInTypes_to_str(builInTypes the_blitype)
  {
    switch(the_blitype)
    {
      ENUM_TO_STR(BuiltInVoid)
      ENUM_TO_STR(BuiltInBool)
      ENUM_TO_STR(BuiltInError)
      default:
      {
        assert(!"Unknown builtInTypes");
        return "UNKNOWN";
      }
    }
  }

  storageClass get_storage_class(const std::string& kind)
  {
    if (kind == "SC_None")
    {
      return SC_None;
    }
    else if (kind == "SC_Extern")
    {
      return SC_Extern;
    }
    else if (kind == "SC_PrivateExtern")
    {
      return SC_PrivateExtern;
    }
    else if (kind == "SC_Static")
    {
      return SC_Static;
    }
    else
    {
      assert(!"Unsupported storage class");
    }
    return SCError;
  }

  const char* storageClass_to_str(storageClass the_strgClass)
  {
    switch(the_strgClass)
    {
      ENUM_TO_STR(SC_None)
      ENUM_TO_STR(SC_Extern)
      ENUM_TO_STR(SC_PrivateExtern)
      ENUM_TO_STR(SC_Static)
      default:
      {
        assert(!"Unknown storageClass");
        return "UNKNOWN";
      }
    }
  }
};
