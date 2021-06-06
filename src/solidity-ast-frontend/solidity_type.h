#ifndef SOLIDITY_TYPE_H_
#define SOLIDITY_TYPE_H_

#include <map>
#include <string>

#define ENUM_TO_STR(s) case s: { return #s; }

namespace SolidityTypes
{
  enum declClass
  {
    DeclLabel = 0, // corresponds to clang::Decl::Label
    DeclFunction,  // corresponds to clang::Decl::Function
    DeclError
  };
  declClass get_decl_class(const std::string& kind);
  const char* declClass_to_str(declClass the_decl);

  // counterparts of type cases of clang::Type::getTypeClass()
  enum typeClass
  {
    TypeBuiltin = 0, // corresponds to clang::Type::Builtin
    TypeError
  };
  typeClass get_type_class(const std::string& kind);
  const char* typeClass_to_str(typeClass the_type);

  // counterparts of type cases of clang::BuiltinType::getKind()
  enum builInTypes
  {
    BuiltInVoid = 0,
    BuiltInBool
  };


};

#endif /* SOLIDITY_TYPE_H_ */
