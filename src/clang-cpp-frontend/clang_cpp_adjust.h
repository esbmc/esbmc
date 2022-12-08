#ifndef CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_H_
#define CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_H_

#include <clang-c-frontend/clang_c_adjust.h>

/**
 * clang C++ adjuster class for:
 *  - symbol adjustment, dealing with ESBMC-IR `symbolt`
 *  - type adjustment, dealing with ESBMC-IR `typet` or other IRs derived from typet
 *  - expression adjustment, dealing with ESBMC-IR `exprt` or other IRs derived from exprt
 *  - code adjustment, dealing with ESBMC-IR `codet` or other IRs derived from codet
 */
class clang_cpp_adjust : public clang_c_adjust
{
public:
  explicit clang_cpp_adjust(contextt &_context);
  virtual ~clang_cpp_adjust() = default;

  bool adjust() override;

  /**
   * methods for type (typet) adjustment
   */
  void adjust_class_type(typet &type);

  /**
   * methods for expression (exprt) adjustment
   * and other IRs derived from exprt
   */
  void adjust_expr(exprt &expr) override;
  void adjust_new(exprt &expr);
  void adjust_side_effect(side_effect_exprt &expr) override;
  void adjust_ptrmember(exprt &expr);
  void adjust_cpp_this(exprt &expr);
  void adjust_cpp_already_checked(exprt &expr);
  // generate member type based on the information in `member` expr
  void gen_member_type(exprt &expr);
  bool get_component(
    const locationt &location,
    const exprt &object,
    const irep_idt &component_name,
    exprt &member);

  /**
   * methods for code (codet) adjustment
   * and other IRs derived from codet
   */
  void adjust_ifthenelse(codet &code) override;
  void adjust_while(codet &code) override;
  void adjust_switch(codet &code) override;
  void adjust_for(codet &code) override;
  void adjust_decl_block(codet &code) override;
  void adjust_assign(codet &code) override;
  void adjust_code_block(codet &code) override;

  /**
   * methods to generate code
   * In some cases, we need to generate additional code for C++, e.g.
   * generate the implicit code for ctors/dtors
   */
  void gen_vtables_dtors(
    const symbolt &symb,
    code_blockt &vtables,
    code_blockt &dtors,
    codet &code);
  codet gen_cpp_destructor(
    const locationt &location,
    const typet &type,
    const exprt &object);

  /**
   * other helper methods
   */
  bool cpp_is_pod(const typet &type) const;
};

#endif /* CLANG_CPP_FRONTEND_CLANG_CPP_ADJUST_H_ */
