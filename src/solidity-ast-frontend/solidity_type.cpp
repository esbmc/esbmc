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
};
