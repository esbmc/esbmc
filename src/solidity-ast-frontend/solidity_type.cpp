#include <solidity-ast-frontend/solidity_type.h>

namespace SolidityTypes
{
  declKind get_decl_kind(const std::string& kind)
  {
    if (kind == "VariableDeclaration")
    {
      return DeclVar;
    }
    else if (kind == "FunctionDefinition")
    {
      return DeclFunction;
    }
    else
    {
      assert(!"Unsupported declKind");
    }
    return DeclKindError;
  }

  const char* declKind_to_str(declKind the_decl)
  {
    switch(the_decl)
    {
      ENUM_TO_STR(DeclVar)
      ENUM_TO_STR(DeclFunction)
      ENUM_TO_STR(DeclKindError)
      default:
      {
        assert(!"Unknown declKind");
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

  builInTypesKind get_builtin_type(const std::string& kind)
  {
    if (kind == "uint8")
    {
      return BuiltInUChar;
    }
    else if (kind == "void")
    {
      return BuiltinVoid;
    }
    else
    {
      assert(!"Unsupported builtInTypes");
    }
    return BuiltInError;
  }
  const char* builInTypesKind_to_str(builInTypesKind the_blitype)
  {
    switch(the_blitype)
    {
      ENUM_TO_STR(BuiltInUChar)
      ENUM_TO_STR(BuiltinVoid)
      ENUM_TO_STR(BuiltinInt)
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
      ENUM_TO_STR(SCError)
      default:
      {
        assert(!"Unknown storageClass");
        return "UNKNOWN";
      }
    }
  }

  const char* stmtClass_to_str(stmtClass the_stmtClass)
  {
    switch(the_stmtClass)
    {
      ENUM_TO_STR(CompoundStmtClass)
      ENUM_TO_STR(BinaryOperatorClass)
      ENUM_TO_STR(DeclRefExprClass)
      ENUM_TO_STR(ImplicitCastExprClass)
      ENUM_TO_STR(IntegerLiteralClass)
      ENUM_TO_STR(StmtClassError)
      default:
      {
        assert(!"Unknown stmtClass");
        return "UNKNOWN";
      }
    }
  }

  binaryOpClass get_binary_op_class(const std::string& kind)
  {
    if (kind == "=")
    {
      return BO_Assign;
    }
    else if (kind == "+")
    {
      return BO_Add;
    }
    else
    {
      assert(!"Unsupported binary operator class");
    }
    return BOError;
  }

  const char* binaryOpClass_to_str(binaryOpClass the_boClass)
  {
    switch(the_boClass)
    {
      ENUM_TO_STR(BO_Assign)
      ENUM_TO_STR(BOError)
      default:
      {
        assert(!"Unknown binaryOpClass");
        return "UNKNOWN";
      }
    }
  }

  const char* declRefKind_to_str(declRefKind the_declRefKind)
  {
    switch(the_declRefKind)
    {
      ENUM_TO_STR(EnumConstantDecl)
      ENUM_TO_STR(ValueDecl)
      ENUM_TO_STR(declRefError)
      default:
      {
        assert(!"Unknown declRefKind");
        return "UNKNOWN";
      }
    }
  }

  const char* castKind_to_str(castKind the_castKind)
  {
    switch(the_castKind)
    {
      ENUM_TO_STR(CK_IntegralCast)
      ENUM_TO_STR(castKindError)
      default:
      {
        assert(!"Unknown castKind");
        return "UNKNOWN";
      }
    }
  }
};
