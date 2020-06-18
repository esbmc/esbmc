#ifndef CLANG_CPP_FRONTEND_CLANG_CPP_CONVERT_H_
#define CLANG_CPP_FRONTEND_CLANG_CPP_CONVERT_H_

#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

#include <clang-c-frontend/clang_c_convert.h>

class clang_cpp_convertert : public clang_c_convertert
{
public:
  clang_cpp_convertert(
    contextt &_context,
    std::vector<std::unique_ptr<clang::ASTUnit>> &_ASTs);
  virtual ~clang_cpp_convertert() = default;

protected:
  bool get_decl(
    const clang::Decl &decl,
    exprt &new_expr) override;

  bool get_struct_union_class_fields(
    const clang::RecordDecl &recordd,
    struct_union_typet &type) override;
};

#endif /* CLANG_C_FRONTEND_CLANG_C_CONVERT_H_ */
