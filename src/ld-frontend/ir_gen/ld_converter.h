#pragma once

#include <ld-frontend/ir/ld_ir.h>
#include <util/context.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <string>

// ld_converter translates LdIR into ESBMC's GOTO IR (irep2 / contextt).
//
// Pattern mirrors python_converter: the caller (ld_languaget::typecheck) passes
// the context in; convert() adds all symbolt entries, the scan loop function,
// and the loop-free __ESBMC_main that calls it.  No C file is produced.
class ld_converter
{
public:
  ld_converter(contextt &context, const LdIR &ir);

  /// Build the symbol table, ld::scan_loop, and __ESBMC_main skeleton.
  /// Throws std::runtime_error on internal errors.
  void convert();

  /// Prepend static-lifetime initialisation assignments to __ESBMC_main.
  /// Call this after all symbols (including those from property_encoder) have
  /// been added to the context.
  void prepend_static_init();

  /// Fault-injection mode: negate contact polarities / degrade coil
  /// assignments.  Used by WP1 validation to plant known semantic errors.
  void enable_fault_injection(bool enabled)
  {
    fault_injection_ = enabled;
  }

private:
  contextt &context_;
  const LdIR &ir_;
  bool fault_injection_ = false;

  typet bool_t() const;
  typet int32_t_() const;
  exprt int_const(long long value) const;

  symbol_exprt declare_variable(const VarDecl &v);
  symbol_exprt declare_bool_shadow(const std::string &id);
  symbol_exprt declare_scoped(const std::string &id, const typet &t);
  symbol_exprt var_expr(const std::string &name) const;

  code_blockt build_scan_body(const exprt &pf_in);

  codet translate_contact(const LdIRNode &n, const exprt &pf_in, exprt &pf_out);
  codet translate_coil(const LdIRNode &n, const exprt &pf);
  codet translate_timer(const LdIRNode &n);
  codet translate_counter(const LdIRNode &n);
  codet translate_arith(const LdIRNode &n);
  codet translate_user_fb(const UserFBExec &ex);

  void emit_scan_function(const code_blockt &scan_body);
  void emit_main_function();
};
