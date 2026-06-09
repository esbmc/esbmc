#pragma once

#include <ld-frontend/ir/ld_ir.h>
#include <util/context.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <string>

// ld_converter translates LdIR into ESBMC's GOTO IR (irep2 / contextt).
//
// Pattern mirrors python_converter: the caller (ld_languaget::typecheck) passes
// the context in; convert() adds all symbolt entries and the scan-loop function
// body.  No C file is produced at any stage.
class ld_converter
{
public:
  ld_converter(contextt &context, const LdIR &ir);

  // Build the full symbol table + scan-loop GOTO function and insert them
  // into the context.  Throws std::runtime_error on internal inconsistencies.
  void convert();

  // Fault-injection mode: negate selected contact polarities / skip coil
  // assignments.  Used by WP1 validation to plant known semantic errors.
  void enable_fault_injection(bool enabled)
  {
    fault_injection_ = enabled;
  }

private:
  contextt &context_;
  const LdIR &ir_;
  bool fault_injection_ = false;

  // Type helpers
  typet bool_t() const;
  typet int32_t_() const;

  // Build a symbolt for an LD variable and insert it into the context.
  // Returns a symbol_exprt referencing the newly created symbol.
  symbol_exprt declare_variable(const VarDecl &v);

  // Retrieve a symbol_exprt for a variable that has already been declared.
  symbol_exprt var_expr(const std::string &name) const;

  // Emit a literal integer constant of the type used for timer/counter fields.
  exprt int_const(long long value) const;

  // Build the scan-loop body (a code_blockt) from the rungs.
  code_blockt build_scan_body(const exprt &pf_in);

  // Per-node translation (one method per SOS rule family)
  codet translate_contact(const LdIRNode &n, const exprt &pf_in, exprt &pf_out);
  codet translate_coil(const LdIRNode &n, const exprt &pf);
  codet translate_timer(const LdIRNode &n);
  codet translate_counter(const LdIRNode &n);
  codet translate_arith(const LdIRNode &n);

  // Build and insert the main scan-loop function into the context.
  void emit_main_function(const code_blockt &scan_body);
};
