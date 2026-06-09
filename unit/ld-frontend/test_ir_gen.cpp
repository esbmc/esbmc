#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <ld-frontend/ir/ld_ir.h>
#include <ld-frontend/ir_gen/ld_converter.h>
#include <util/context.h>
#include <util/std_code.h>
#include <util/std_expr.h>

static LdIR make_empty_ir()
{
  LdIR ir;
  ir.source_file = "<test>";
  return ir;
}

static void add_bool_var(LdIR &ir, const std::string &name)
{
  VarDecl v;
  v.name = name;
  v.kind = VarKind::BOOL;
  ir.variables.push_back(v);
}

static void add_int_var(LdIR &ir, const std::string &name)
{
  VarDecl v;
  v.name = name;
  v.kind = VarKind::INT;
  ir.variables.push_back(v);
}

TEST_CASE("__ESBMC_main is loop-free; ld::scan_loop holds the while loop", "[ir_gen]")
{
  LdIR ir = make_empty_ir();
  add_bool_var(ir, "A");

  contextt ctx;
  ld_converter conv(ctx, ir);
  conv.convert();
  conv.prepend_static_init();

  const symbolt *main_sym = ctx.find_symbol("__ESBMC_main");
  REQUIRE(main_sym != nullptr);

  bool has_while = false;
  for (const auto &op : main_sym->get_value().operands())
    if (static_cast<const codet &>(op).get_statement() == "while")
      has_while = true;
  REQUIRE(!has_while);

  const symbolt *scan_sym = ctx.find_symbol("ld::scan_loop");
  REQUIRE(scan_sym != nullptr);
  const exprt &scan_body = scan_sym->get_value();
  REQUIRE(!scan_body.operands().empty());
  REQUIRE(
    static_cast<const codet &>(scan_body.operands().front()).get_statement() ==
    "while");
}

TEST_CASE("all LD symbols have mode == \"LD\"", "[ir_gen]")
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
      REQUIRE(s.mode == "LD");
  });
}

TEST_CASE("CTU counter declares per-instance shadow variable", "[ir_gen]")
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

  const symbolt *prev = ctx.find_symbol("ld::__ctr_prev_CTU1");
  REQUIRE(prev != nullptr);
  REQUIRE(prev->get_type() == typet("bool"));
  REQUIRE(prev->mode == "LD");
  REQUIRE(prev->name == "__ctr_prev_CTU1");
}

TEST_CASE("CTD counter declares per-instance shadow variable", "[ir_gen]")
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

  const symbolt *prev = ctx.find_symbol("ld::__ctr_prev_CTD1");
  REQUIRE(prev != nullptr);
  REQUIRE(prev->get_type() == typet("bool"));
  REQUIRE(prev->mode == "LD");
  REQUIRE(prev->name == "__ctr_prev_CTD1");
}

TEST_CASE("two CTU instances sharing a CU signal get independent shadow variables", "[ir_gen]")
{
  LdIR ir = make_empty_ir();
  add_bool_var(ir, "CU_shared");
  add_bool_var(ir, "Q1");
  add_bool_var(ir, "Q2");
  add_int_var(ir, "CV1");
  add_int_var(ir, "CV2");

  auto make_ctu = [](const std::string &inst, const std::string &cv,
                     const std::string &q) {
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
  REQUIRE(sym_a != nullptr);
  REQUIRE(sym_b != nullptr);
  REQUIRE(sym_a->name == "__ctr_prev_CTU_A");
  REQUIRE(sym_b->name == "__ctr_prev_CTU_B");
}
