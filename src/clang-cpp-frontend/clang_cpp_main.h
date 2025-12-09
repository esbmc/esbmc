#ifndef CLANG_CPP_FRONTEND_CLANG_CPP_MAIN_H_
#define CLANG_CPP_FRONTEND_CLANG_CPP_MAIN_H_

#include <clang-c-frontend/clang_c_main.h>

class clang_cpp_maint : public clang_c_maint
{
public:
  clang_cpp_maint(contextt &_context);

  // code adjustment for C++, e.g. adding implicit this in ctor when
  // adjusting the object initialization
  void adjust_init(code_assignt &assignment, codet &adjusted) override;
  void convert_expression_to_code(exprt &expr);
};

#endif /* CLANG_CPP_FRONTEND_CLANG_CPP_MAIN_H_ */
