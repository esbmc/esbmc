#ifndef SOLIDITY_TYPE_H_
#define SOLIDITY_TYPE_H_

namespace solidityTypes
{
  // counterparts of type cases of clang::Type::getTypeClass()
  enum typeClass
  {
    TypeBuiltin = 0,
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
