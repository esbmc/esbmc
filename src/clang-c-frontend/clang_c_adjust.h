#ifndef CLANG_C_FRONTEND_CLANG_C_ADJUST_H_
#define CLANG_C_FRONTEND_CLANG_C_ADJUST_H_

#include <util/context.h>
#include <util/namespace.h>
#include <util/std_code.h>
#include <util/std_expr.h>

/**
 * clang C adjuster class for:
 *  - symbol adjustment, dealing with ESBMC-IR `symbolt`
 *  - type adjustment, dealing with ESBMC-IR `typet` or other IRs derived from typet
 *  - expression adjustment, dealing with ESBMC-IR `exprt` or other IRs derived from exprt
 *  - code adjustment, dealing with ESBMC-IR `codet` or other IRs derived from codet
 */
class clang_c_adjust
{
public:
  explicit clang_c_adjust(contextt &_context);
  virtual ~clang_c_adjust() = default;

  bool adjust();

protected:
  contextt &context;
  namespacet ns;

  /**
   * methods for symbol adjustment
   */
  virtual void adjust_symbol(symbolt &symbol);
  void adjust_argc_argv(const symbolt &main_symbol);

  /**
   * methods for type (typet) adjustment
   */
  void adjust_type(typet &type);

  /**
   * methods for expression (exprt) adjustment
   * and other IRs derived from exprt
   */
  void adjust_expr(exprt &expr);
  void adjust_side_effect_assignment(exprt &expr);
  void adjust_side_effect_function_call(side_effect_expr_function_callt &expr);
  void adjust_side_effect_statement_expression(side_effect_exprt &expr);
  virtual void adjust_member(member_exprt &expr);
  void adjust_expr_binary_arithmetic(exprt &expr);
  void adjust_expr_shifts(exprt &expr);
  void adjust_expr_unary_boolean(exprt &expr);
  void adjust_expr_binary_boolean(exprt &expr);
  virtual void adjust_expr_rel(exprt &expr);
  void adjust_float_arith(exprt &expr);
  void adjust_index(index_exprt &index);
  void adjust_dereference(exprt &deref);
  void adjust_address_of(exprt &expr);
  void adjust_sizeof(exprt &expr);
  virtual void adjust_side_effect(side_effect_exprt &expr);
  void adjust_symbol(exprt &expr);
  void adjust_comma(exprt &expr);
  void adjust_builtin_va_arg(exprt &expr);
  virtual void
  adjust_function_call_arguments(side_effect_expr_function_callt &expr);
  void do_special_functions(side_effect_expr_function_callt &expr);
  void adjust_operands(exprt &expr);
  virtual void adjust_if(exprt &expr);

  /**
   * methods for code (codet) adjustment
   * and other IRs derived from codet
   */
  void adjust_code(codet &code);
  virtual void adjust_ifthenelse(codet &code);
  virtual void adjust_while(codet &code);
  virtual void adjust_for(codet &code);
  virtual void adjust_switch(codet &code);
  void adjust_assign(codet &code);
  void adjust_decl(codet &code);
  // For class instantiation in C++, we need to adjust the side-effect of constructor
  virtual void adjust_decl_block(codet &code);

  exprt is_gcc_polymorphic_builtin(
    const irep_idt &identifier,
    const exprt::operandst &arguments);

  code_blockt instantiate_gcc_polymorphic_builtin(
    const irep_idt &identifier,
    const symbol_exprt &function_symbol);

  /**
   * ancillary methods to support the expr/code adjustments above
   */
  virtual void align_se_function_call_return_type(
    exprt &f_op,
    side_effect_expr_function_callt &expr);
};

#endif /* CLANG_C_FRONTEND_CLANG_C_ADJUST_H_ */
