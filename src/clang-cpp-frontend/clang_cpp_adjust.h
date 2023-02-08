#ifndef CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_H_
#define CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_H_

#include <clang-c-frontend/clang_c_adjust.h>

/**
 * clang C++ adjuster class for:
 *  - expression adjustment, dealing with ESBMC-IR `exprt` or other IRs derived from exprt
 *  - code adjustment, dealing with ESBMC-IR `codet` or other IRs derived from codet
 */
class clang_cpp_adjust : public clang_c_adjust
{
public:
  explicit clang_cpp_adjust(contextt &_context);
  virtual ~clang_cpp_adjust() = default;

  /**
   * methods for code (codet) adjustment
   * and other IRs derived from codet
   */
  void adjust_while(codet &code) override;
  void adjust_switch(codet &code) override;
  void adjust_for(codet &code) override;
  void adjust_ifthenelse(codet &code) override;
  void adjust_decl_block(codet &code) override;

  /**
   * methods for expression (exprt) adjustment
   * and other IRs derived from exprt
   */
  void adjust_member(member_exprt &expr) override;
  void adjust_side_effect(side_effect_exprt &expr) override;
  void adjust_new(exprt &expr);
  void adjust_struct_method_call(member_exprt &expr);
};

#endif /* CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_H_ */
