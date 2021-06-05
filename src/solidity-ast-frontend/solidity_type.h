#ifndef SOLIDITY_TYPE_H_
#define SOLIDITY_TYPE_H_

#include <map>
#include <string>

namespace SolidityTypes
{
  enum declClass
  {
    DeclLabel = 0, // corresponds to clang::Decl::Label
    DeclFunction,  // corresponds to clang::Decl::Function
    DeclError
  };
  declClass get_decl_class(const std::string& kind);

  // counterparts of type cases of clang::Type::getTypeClass()
  enum typeClass
  {
    TypeBuiltin = 0, // corresponds to clang::Type::Builtin
    TypeParen
  };

  // counterparts of type cases of clang::BuiltinType::getKind()
  enum builInTypes
  {
    BuiltInVoid = 0,
    BuiltInBool
  };


};

#endif /* SOLIDITY_TYPE_H_ */
