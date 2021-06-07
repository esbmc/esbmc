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

  // counterparts of clang::Type::getTypeClass() used by get_type during conversion
  enum typeClass
  {
    TypeBuiltin = 0, // corresponds to clang::Type::Builtin
    TypeError
  };
  typeClass get_type_class(const std::string& kind);
  const char* typeClass_to_str(typeClass the_type);

  // counterparts of clang::BuiltinType::getKind() used by get_builtin_type during conversion
  enum builInTypes
  {
    BuiltInVoid = 0,
    BuiltInError
  };
  builInTypes get_builtin_type(const std::string& kind);
  const char* builInTypes_to_str(builInTypes the_blitype);

  enum storageClass
  {
    SC_None = 0,
    SC_Extern,
    SC_PrivateExtern,
    SC_Static,
    SCError
  };
  storageClass get_storage_class(const std::string& kind);
  const char* storageClass_to_str(storageClass the_strgClass);
};

#endif /* SOLIDITY_TYPE_H_ */
