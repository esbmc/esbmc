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

  /// Build an integer arithmetic expression (plus/minus/mult/div) carrying an
  /// explicit int32 result type.  The LD front-end emits GOTO IR with no adjust
  /// pass, so an untyped result (as left by the two-operand plus_exprt/mult_exprt
  /// constructors) would abort the irep2 migration on a width-consistency check.
  exprt int_arith(const irep_idt &op, const exprt &a, const exprt &b) const;

  symbol_exprt declare_variable(const VarDecl &v);
  symbol_exprt declare_bool_shadow(const std::string &id);
  symbol_exprt var_expr(const std::string &name) const;

  code_blockt build_scan_body(const exprt &pf_in);

  /// READ_INPUTS phase: assign a fresh nondeterministic value to every input
  /// variable at the start of each scan cycle, modelling the physical capture
  /// of plant inputs.  Without this the scan loop would only ever explore the
  /// all-zero input trace, making every safety proof vacuous.
  /// Inputs are re-sampled each cycle and assumed not to be coil-driven; if a
  /// variable were both an input and a coil target, re-sampling over-approximates
  /// its value, which stays sound for safety (it never hides a violation).
  code_blockt build_read_inputs() const;

  codet translate_contact(const LdIRNode &n, const exprt &pf_in, exprt &pf_out);
  codet translate_coil(const LdIRNode &n, const exprt &pf);
  codet translate_timer(const LdIRNode &n);
  codet translate_counter(const LdIRNode &n);
  codet translate_arith(const LdIRNode &n);

  void emit_scan_function(const code_blockt &scan_body);
  void emit_main_function();
};
