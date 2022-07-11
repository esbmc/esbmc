#ifndef CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_H_
#define CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_H_

#include <clang-c-frontend/clang_c_adjust.h>

class clang_cpp_adjust : public clang_c_adjust
{
public:
  explicit clang_cpp_adjust(contextt &_context);
  virtual ~clang_cpp_adjust() = default;

  void adjust_ifthenelse(codet &code) override;
  void adjust_while(codet &code) override;
  void adjust_switch(codet &code) override;
  void adjust_for(codet &code) override;
  void adjust_decl_block(codet &code) override;

  void adjust_side_effect(side_effect_exprt &expr) override;

  void adjust_new(exprt &expr);
};

#endif /* CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_H_ */
