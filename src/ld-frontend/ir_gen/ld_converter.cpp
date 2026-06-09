#include <ld-frontend/ir_gen/ld_converter.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/symbol.h>
#include <stdexcept>

ld_converter::ld_converter(contextt &context, const LdIR &ir)
  : context_(context), ir_(ir)
{}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

typet ld_converter::bool_t() const
{
  return typet("bool");
}

typet ld_converter::int32_t_() const
{
  return int_type();
}

exprt ld_converter::int_const(long long value) const
{
  return from_integer(BigInt(value), int32_t_());
}

// Prefix all LD variable names to avoid clashes with C runtime symbols.
static std::string ld_name(const std::string &var)
{
  return "ld::" + var;
}

symbol_exprt ld_converter::declare_variable(const VarDecl &v)
{
  symbolt sym;
  sym.id = ld_name(v.name);
  sym.name = v.name;  // human-readable short name
  sym.module = "ld";
  sym.lvalue = true;
  sym.static_lifetime = true;
  sym.file_local = false;
  sym.is_extern = false;

  locationt loc;
  loc.set_file(v.loc.file);
  loc.set_line(v.loc.line);
  loc.set_column(v.loc.col);
  sym.location = loc;

  switch (v.kind)
  {
  case VarKind::BOOL:
    sym.set_type(bool_t());
    sym.set_value(false_exprt());
    break;
  case VarKind::INT:
  case VarKind::DINT:
  case VarKind::TIME:
    sym.set_type(int32_t_());
    sym.set_value(int_const(0));
    break;
  }

  context_.move_symbol_to_context(sym);
  return symbol_exprt(ld_name(v.name), sym.get_type());
}

symbol_exprt ld_converter::var_expr(const std::string &name) const
{
  const symbolt *sym = context_.find_symbol(ld_name(name));
  if (!sym)
    throw std::runtime_error("ld_converter: undeclared variable '" + name + "'");
  return symbol_exprt(ld_name(name), sym->get_type());
}

// -----------------------------------------------------------------------
// Per-node translation
// -----------------------------------------------------------------------

// ContactEval: power-flow out = pf_in AND (var | !var)
// Returns the new pf_out expression via out-parameter.
codet ld_converter::translate_contact(
  const LdIRNode &n,
  const exprt &pf_in,
  exprt &pf_out)
{
  symbol_exprt var = var_expr(n.variable);
  exprt contact_val;
  ContactKind eff_kind = n.contact_kind;
  if (fault_injection_)
    eff_kind = (eff_kind == ContactKind::NormallyOpen)
                 ? ContactKind::NormallyClosed
                 : ContactKind::NormallyOpen;

  if (eff_kind == ContactKind::NormallyClosed)
    contact_val = not_exprt(var);
  else
    contact_val = var;

  pf_out = and_exprt(pf_in, contact_val);

  // Contact evaluation is purely combinatorial — no instructions emitted.
  return code_skipt();
}

// CoilAssign: assign coil variable according to power-flow
codet ld_converter::translate_coil(const LdIRNode &n, const exprt &pf)
{
  symbol_exprt var = var_expr(n.variable);
  CoilKind eff_kind = n.coil_kind;
  if (fault_injection_)
    eff_kind = CoilKind::Output;

  code_blockt blk;
  switch (eff_kind)
  {
  case CoilKind::Output:
    blk.copy_to_operands(code_assignt(var, pf));
    break;
  case CoilKind::Set:
  {
    code_ifthenelset ite;
    ite.cond() = pf;
    ite.then_case() = code_assignt(var, true_exprt());
    blk.copy_to_operands(ite);
    break;
  }
  case CoilKind::Reset:
  {
    code_ifthenelset ite;
    ite.cond() = pf;
    ite.then_case() = code_assignt(var, false_exprt());
    blk.copy_to_operands(ite);
    break;
  }
  }
  return blk;
}

// TimerStep: synchronous fixed-tick model (§3.3 of the implementation plan)
//   TON: if IN then ET++ else ET:=0;  Q := (ET >= PT)
//   TOF: if !IN then ET++ else ET:=0; Q := (ET < PT)
//   TP:  simplified to TON semantics; full TP spec in T1.2
codet ld_converter::translate_timer(const LdIRNode &n)
{
  symbol_exprt et_sym = var_expr(n.timer_ET);
  symbol_exprt pt_sym = var_expr(n.timer_PT);
  symbol_exprt q_sym  = var_expr(n.timer_Q);
  symbol_exprt in_sym = var_expr(n.timer_IN);

  exprt one  = gen_one(int32_t_());
  exprt zero = gen_zero(int32_t_());

  // if (IN|!IN) then ET := ET+1 else ET := 0
  exprt condition;
  if (n.timer_kind == FBKind::TOF)
    condition = not_exprt(in_sym);
  else
    condition = in_sym;

  code_ifthenelset et_step;
  et_step.cond()       = condition;
  et_step.then_case()  = code_assignt(et_sym, plus_exprt(et_sym, one));
  et_step.else_case()  = code_assignt(et_sym, zero);

  // Q := (ET >= PT)  for TON/TP;  Q := (ET < PT) for TOF
  exprt q_expr;
  if (n.timer_kind == FBKind::TOF)
    q_expr = binary_relation_exprt(et_sym, "<", pt_sym);
  else
    q_expr = binary_relation_exprt(et_sym, ">=", pt_sym);

  code_blockt blk;
  blk.copy_to_operands(et_step);
  blk.copy_to_operands(code_assignt(q_sym, q_expr));
  return blk;
}

// CounterStep: per-scan step
//   CTU: if CU then CV++;  Q := (CV >= PV);  if R then CV:=0
//   CTD: if CD then CV--;  Q := (CV <= 0)
codet ld_converter::translate_counter(const LdIRNode &n)
{
  code_blockt blk;
  exprt one  = gen_one(int32_t_());
  exprt zero = gen_zero(int32_t_());

  if (n.ctr_kind == FBKind::CTU)
  {
    symbol_exprt cu  = var_expr(n.ctr_CU);
    symbol_exprt cv  = var_expr(n.ctr_CV);
    symbol_exprt pv  = var_expr(n.ctr_PV);
    symbol_exprt q   = var_expr(n.ctr_Q);

    code_ifthenelset cu_step;
    cu_step.cond()      = cu;
    cu_step.then_case() = code_assignt(cv, plus_exprt(cv, one));
    blk.copy_to_operands(cu_step);
    blk.copy_to_operands(code_assignt(q, binary_relation_exprt(cv, ">=", pv)));

    if (!n.ctr_R.empty())
    {
      symbol_exprt r = var_expr(n.ctr_R);
      code_ifthenelset r_step;
      r_step.cond()      = r;
      r_step.then_case() = code_assignt(cv, zero);
      blk.copy_to_operands(r_step);
    }
  }
  else // CTD
  {
    symbol_exprt cd  = var_expr(n.ctr_CD);
    symbol_exprt cv  = var_expr(n.ctr_CV);
    symbol_exprt q   = var_expr(n.ctr_Q);
    exprt neg_one = from_integer(BigInt(-1), int32_t_());

    code_ifthenelset cd_step;
    cd_step.cond()      = cd;
    cd_step.then_case() = code_assignt(cv, plus_exprt(cv, neg_one));
    blk.copy_to_operands(cd_step);
    blk.copy_to_operands(code_assignt(q, binary_relation_exprt(cv, "<=", zero)));
  }
  return blk;
}

// ArithStep: OUT := IN1 op IN2
codet ld_converter::translate_arith(const LdIRNode &n)
{
  symbol_exprt in1 = var_expr(n.arith_IN1);
  symbol_exprt in2 = var_expr(n.arith_IN2);
  symbol_exprt out = var_expr(n.arith_OUT);

  exprt op_expr;
  switch (n.arith_kind)
  {
  case FBKind::ADD:
    op_expr = plus_exprt(in1, in2);
    break;
  case FBKind::SUB:
    op_expr = exprt(exprt::minus, int32_t_());
    op_expr.copy_to_operands(in1, in2);
    break;
  case FBKind::MUL:
    op_expr = mult_exprt(in1, in2);
    break;
  case FBKind::DIV:
    op_expr = exprt(exprt::div, int32_t_());
    op_expr.copy_to_operands(in1, in2);
    break;
  case FBKind::MOVE:
    op_expr = in1;
    break;
  default:
    op_expr = in1;
    break;
  }
  return code_assignt(out, op_expr);
}

// -----------------------------------------------------------------------
// Scan body construction
// -----------------------------------------------------------------------

code_blockt ld_converter::build_scan_body(const exprt &)
{
  code_blockt scan_body;

  for (const auto &rung : ir_.rungs)
  {
    code_blockt rung_blk;
    exprt rung_pf = true_exprt(); // power-flow starts true at each rung

    for (const auto &node : rung.nodes)
    {
      switch (node.kind)
      {
      case LdIRNodeKind::ContactEval:
      {
        exprt pf_out;
        codet c = translate_contact(node, rung_pf, pf_out);
        if (c.get_statement() != "skip")
          rung_blk.copy_to_operands(c);
        rung_pf = pf_out;
        break;
      }
      case LdIRNodeKind::CoilAssign:
      {
        codet c = translate_coil(node, rung_pf);
        rung_blk.move_to_operands(c);
        break;
      }
      case LdIRNodeKind::TimerStep:
      {
        codet c = translate_timer(node);
        rung_blk.move_to_operands(c);
        break;
      }
      case LdIRNodeKind::CounterStep:
      {
        codet c = translate_counter(node);
        rung_blk.move_to_operands(c);
        break;
      }
      case LdIRNodeKind::ArithStep:
      {
        codet c = translate_arith(node);
        rung_blk.move_to_operands(c);
        break;
      }
      }
    }

    scan_body.move_to_operands(rung_blk);
  }

  return scan_body;
}

// -----------------------------------------------------------------------
// Main function emission
// -----------------------------------------------------------------------

void ld_converter::emit_main_function(const code_blockt &scan_body)
{
  code_typet main_type;
  main_type.return_type() = empty_typet();

  symbolt main_sym;
  main_sym.id = "__ESBMC_main";
  main_sym.name = "__ESBMC_main";
  main_sym.module = "ld";
  main_sym.set_type(main_type);
  main_sym.lvalue = false;
  main_sym.is_extern = false;
  main_sym.file_local = false;
  main_sym.static_lifetime = false;

  locationt loc;
  loc.set_file(ir_.source_file);
  main_sym.location = loc;

  code_blockt main_body;

  // Initialise all static-lifetime variables before entering the scan loop.
  context_.foreach_operand_in_order([&main_body](const symbolt &s) {
    if (s.static_lifetime && !s.get_value().is_nil() && !s.get_type().is_code())
    {
      code_assignt assign(symbol_expr(s), s.get_value());
      assign.location() = s.location;
      main_body.copy_to_operands(assign);
    }
  });

  // The scan loop: while(true) { scan_body }
  code_whilet scan_loop;
  scan_loop.cond() = true_exprt();
  scan_loop.body() = scan_body;
  main_body.copy_to_operands(scan_loop);

  main_sym.set_value(main_body);
  context_.move_symbol_to_context(main_sym);
}

// -----------------------------------------------------------------------
// Top-level convert()
// -----------------------------------------------------------------------

void ld_converter::convert()
{
  // 1. Declare all LD variables in the symbol table.
  for (const auto &v : ir_.variables)
    declare_variable(v);

  // 2. Build the scan-loop body.
  code_blockt scan_body = build_scan_body(true_exprt());

  // 3. Emit the __ESBMC_main function.
  emit_main_function(scan_body);
}
