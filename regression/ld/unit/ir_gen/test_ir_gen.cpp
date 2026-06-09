// Unit tests for ld_converter.
// Standalone; return 0 on pass.

#include <ld-frontend/ir/ld_ir.h>
#include <ld-frontend/ir_gen/ld_converter.h>
#include <util/context.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <cassert>
#include <iostream>

// Build a minimal LdIR with one variable and no rungs.
static LdIR make_empty_ir(const std::string &source = "<test>")
{
  LdIR ir;
  ir.source_file = source;
  return ir;
}

// Add a BOOL variable to an LdIR.
static void add_bool_var(LdIR &ir, const std::string &name, bool is_input = false)
{
  VarDecl v;
  v.name = name;
  v.kind = VarKind::BOOL;
  v.is_input = is_input;
  ir.variables.push_back(v);
}

// Add an INT variable to an LdIR.
static void add_int_var(LdIR &ir, const std::string &name)
{
  VarDecl v;
  v.name = name;
  v.kind = VarKind::INT;
  ir.variables.push_back(v);
}

// Fix 11: __ESBMC_main must not contain a while loop (loop is in ld::scan_loop).
static void test_main_function_is_loop_free()
{
  LdIR ir = make_empty_ir();
  add_bool_var(ir, "A");

  contextt ctx;
  ld_converter conv(ctx, ir);
  conv.convert();
  conv.prepend_static_init();

  const symbolt *main_sym = ctx.find_symbol("__ESBMC_main");
  assert(main_sym != nullptr);

  // Walk the body of __ESBMC_main looking for a while statement.
  const exprt &body = main_sym->get_value();
  bool has_while = false;
  for (const auto &op : body.operands())
  {
    const codet &stmt = static_cast<const codet &>(op);
    if (stmt.get_statement() == "while")
      has_while = true;
  }
  assert(!has_while);

  // ld::scan_loop must exist and must contain the while loop.
  const symbolt *scan_sym = ctx.find_symbol("ld::scan_loop");
  assert(scan_sym != nullptr);
  const exprt &scan_body = scan_sym->get_value();
  assert(!scan_body.operands().empty());
  const codet &first = static_cast<const codet &>(scan_body.operands().front());
  assert(first.get_statement() == "while");

  std::cout << "PASS: test_main_function_is_loop_free\n";
}

// Fix 16+17: all LD symbols must have mode == "LD".
static void test_symbol_mode_set()
{
  LdIR ir = make_empty_ir();
  add_bool_var(ir, "A");
  add_bool_var(ir, "B");

  contextt ctx;
  ld_converter conv(ctx, ir);
  conv.convert();
  conv.prepend_static_init();

  ctx.foreach_operand_in_order([](const symbolt &s) {
    if (s.module == "ld")
      assert(s.mode == "LD");
  });

  std::cout << "PASS: test_symbol_mode_set\n";
}

// Fix 12: CTU counter must be edge-triggered.
// Build a CTU rung, convert it, inspect that the body uses
// the pattern: if (CU && !CU_prev) { CV++ }; CU_prev := CU.
// We verify this structurally: the shadow symbol ld::__ctr_prev_CTU1 exists.
static void test_counter_edge_shadow_variable_declared()
{
  LdIR ir = make_empty_ir();
  add_bool_var(ir, "CU_sig");
  add_bool_var(ir, "Q_sig");
  add_int_var(ir, "CV_sig");

  LdIRNode node;
  node.kind = LdIRNodeKind::CounterStep;
  node.ctr_kind = FBKind::CTU;
  node.ctr_instance = "CTU1";
  node.ctr_CU = "CU_sig";
  node.ctr_Q = "Q_sig";
  node.ctr_CV = "CV_sig";

  LdIRRung rung;
  rung.id = "1";
  rung.nodes.push_back(node);
  ir.rungs.push_back(rung);

  contextt ctx;
  ld_converter conv(ctx, ir);
  conv.convert();
  conv.prepend_static_init();

  // The shadow variable for edge detection must exist in the context.
  const symbolt *prev_sym = ctx.find_symbol("ld::__ctr_prev_CTU1");
  assert(prev_sym != nullptr);
  assert(prev_sym->get_type() == typet("bool"));
  assert(prev_sym->mode == "LD");
  assert(prev_sym->name == "__ctr_prev_CTU1"); // sym.name is short (no ld:: prefix)
  std::cout << "PASS: test_counter_edge_shadow_variable_declared\n";
}

// Fix 12: CTD counter must also get a shadow variable.
static void test_counter_ctd_shadow_variable_declared()
{
  LdIR ir = make_empty_ir();
  add_bool_var(ir, "CD_sig");
  add_bool_var(ir, "Q_sig");
  add_int_var(ir, "CV_sig");

  LdIRNode node;
  node.kind = LdIRNodeKind::CounterStep;
  node.ctr_kind = FBKind::CTD;
  node.ctr_instance = "CTD1";
  node.ctr_CD = "CD_sig";
  node.ctr_Q = "Q_sig";
  node.ctr_CV = "CV_sig";

  LdIRRung rung;
  rung.id = "1";
  rung.nodes.push_back(node);
  ir.rungs.push_back(rung);

  contextt ctx;
  ld_converter conv(ctx, ir);
  conv.convert();
  conv.prepend_static_init();

  const symbolt *prev_sym = ctx.find_symbol("ld::__ctr_prev_CTD1");
  assert(prev_sym != nullptr);
  assert(prev_sym->get_type() == typet("bool"));
  assert(prev_sym->mode == "LD");
  assert(prev_sym->name == "__ctr_prev_CTD1");
  std::cout << "PASS: test_counter_ctd_shadow_variable_declared\n";
}

// Fix 12: two CTU instances with the same CU signal get independent
// shadow variables (per-instance, not per-signal).
static void test_counter_two_instances_independent_shadows()
{
  LdIR ir = make_empty_ir();
  add_bool_var(ir, "CU_shared");
  add_bool_var(ir, "Q1");
  add_bool_var(ir, "Q2");
  add_int_var(ir, "CV1");
  add_int_var(ir, "CV2");

  auto make_ctu = [](const std::string &inst, const std::string &cv, const std::string &q) {
    LdIRNode n;
    n.kind = LdIRNodeKind::CounterStep;
    n.ctr_kind = FBKind::CTU;
    n.ctr_instance = inst;
    n.ctr_CU = "CU_shared";
    n.ctr_CV = cv;
    n.ctr_Q = q;
    return n;
  };

  LdIRRung rung;
  rung.id = "1";
  rung.nodes.push_back(make_ctu("CTU_A", "CV1", "Q1"));
  rung.nodes.push_back(make_ctu("CTU_B", "CV2", "Q2"));
  ir.rungs.push_back(rung);

  contextt ctx;
  ld_converter conv(ctx, ir);
  conv.convert();
  conv.prepend_static_init();

  const symbolt *sym_a = ctx.find_symbol("ld::__ctr_prev_CTU_A");
  const symbolt *sym_b = ctx.find_symbol("ld::__ctr_prev_CTU_B");
  assert(sym_a != nullptr);
  assert(sym_b != nullptr);
  assert(sym_a->name == "__ctr_prev_CTU_A");
  assert(sym_b->name == "__ctr_prev_CTU_B");
  std::cout << "PASS: test_counter_two_instances_independent_shadows\n";
}

int main()
{
  test_main_function_is_loop_free();
  test_symbol_mode_set();
  test_counter_edge_shadow_variable_declared();
  test_counter_ctd_shadow_variable_declared();
  test_counter_two_instances_independent_shadows();
  std::cout << "All IR generator tests passed.\n";
  return 0;
}
