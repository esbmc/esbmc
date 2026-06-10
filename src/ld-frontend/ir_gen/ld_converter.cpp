#include <ld-frontend/ir_gen/ld_converter.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/symbol.h>
#include <stdexcept>

ld_converter::ld_converter(contextt &context, const LdIR &ir)
  : context_(context), ir_(ir)
{
}

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

static std::string ld_name(const std::string &var)
{
  return "ld::" + var;
}

symbol_exprt ld_converter::declare_variable(const VarDecl &v)
{
  symbolt sym;
  sym.id = ld_name(v.name);
  sym.name = v.name;
  sym.module = "ld";
  sym.mode = "LD";
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

// Declare a BOOL shadow variable for edge detection; no-op if already declared.
symbol_exprt ld_converter::declare_bool_shadow(const std::string &id)
{
  if (!context_.find_symbol(id))
  {
    symbolt sym;
    static const std::string kPrefix = "ld::";
    sym.id = id;
    sym.name = (id.compare(0, kPrefix.size(), kPrefix) == 0)
                 ? id.substr(kPrefix.size())
                 : id;
    sym.module = "ld";
    sym.mode = "LD";
    sym.lvalue = true;
    sym.static_lifetime = true;
    sym.file_local = false;
    sym.is_extern = false;
    sym.set_type(bool_t());
    sym.set_value(false_exprt());
    locationt loc;
    loc.set_file(ir_.source_file);
    sym.location = loc;
    context_.move_symbol_to_context(sym);
  }
  return symbol_exprt(id, bool_t());
}

symbol_exprt ld_converter::var_expr(const std::string &name) const
{
  const symbolt *sym = context_.find_symbol(ld_name(name));
  if (!sym)
    throw std::runtime_error(
      "ld_converter: undeclared variable '" + name + "'");
  return symbol_exprt(ld_name(name), sym->get_type());
}

// -----------------------------------------------------------------------
// Per-node translation
// -----------------------------------------------------------------------

codet ld_converter::translate_contact(
  const LdIRNode &n,
  const exprt &pf_in,
  exprt &pf_out)
{
  symbol_exprt var = var_expr(n.variable);
  ContactKind eff_kind = n.contact_kind;
  if (fault_injection_)
    eff_kind = (eff_kind == ContactKind::NormallyOpen)
                 ? ContactKind::NormallyClosed
                 : ContactKind::NormallyOpen;

  exprt contact_val = (eff_kind == ContactKind::NormallyClosed)
                        ? static_cast<exprt>(not_exprt(var))
                        : static_cast<exprt>(var);
  pf_out = and_exprt(pf_in, contact_val);
  return code_skipt();
}

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

// TimerStep: synchronous fixed-tick model
//   TON: if IN then ET++ else ET:=0;  Q := (ET >= PT)
//   TOF: if !IN then ET++ else ET:=0; Q := (ET < PT)
//   TP:  simplified to TON semantics
codet ld_converter::translate_timer(const LdIRNode &n)
{
  symbol_exprt et_sym = var_expr(n.timer_ET);
  symbol_exprt pt_sym = var_expr(n.timer_PT);
  symbol_exprt q_sym = var_expr(n.timer_Q);
  symbol_exprt in_sym = var_expr(n.timer_IN);

  exprt one = gen_one(int32_t_());
  exprt zero = gen_zero(int32_t_());

  exprt condition = (n.timer_kind == FBKind::TOF) ? not_exprt(in_sym)
                                                  : static_cast<exprt>(in_sym);

  code_ifthenelset et_step;
  et_step.cond() = condition;
  et_step.then_case() = code_assignt(et_sym, plus_exprt(et_sym, one));
  et_step.else_case() = code_assignt(et_sym, zero);

  exprt q_expr =
    (n.timer_kind == FBKind::TOF)
      ? binary_relation_exprt(et_sym, "<", pt_sym)
      : static_cast<exprt>(binary_relation_exprt(et_sym, ">=", pt_sym));

  code_blockt blk;
  blk.copy_to_operands(et_step);
  blk.copy_to_operands(code_assignt(q_sym, q_expr));
  return blk;
}

// CounterStep: IEC 61131-3 §2.5.2.3 — edge-triggered on rising CU/CD.
//   CTU: if (CU && !CU_prev) CV++;  if (R) CV:=0;  CU_prev:=CU; Q:=(CV>=PV)
//   CTD: if (CD && !CD_prev) CV--;  CD_prev:=CD; Q:=(CV<=0)
codet ld_converter::translate_counter(const LdIRNode &n)
{
  code_blockt blk;
  exprt one = gen_one(int32_t_());
  exprt zero = gen_zero(int32_t_());

  if (n.ctr_kind == FBKind::CTU)
  {
    symbol_exprt cu = var_expr(n.ctr_CU);
    symbol_exprt cv = var_expr(n.ctr_CV);
    symbol_exprt q = var_expr(n.ctr_Q);
    symbol_exprt cu_prev =
      declare_bool_shadow(ld_name("__ctr_prev_" + n.ctr_instance));

    code_ifthenelset cu_step;
    cu_step.cond() = and_exprt(cu, not_exprt(cu_prev));
    cu_step.then_case() = code_assignt(cv, plus_exprt(cv, one));
    blk.copy_to_operands(cu_step);

    if (!n.ctr_R.empty())
    {
      symbol_exprt r = var_expr(n.ctr_R);
      code_ifthenelset r_step;
      r_step.cond() = r;
      r_step.then_case() = code_assignt(cv, zero);
      blk.copy_to_operands(r_step);
    }

    blk.copy_to_operands(code_assignt(cu_prev, cu));

    if (!n.ctr_PV.empty())
      blk.copy_to_operands(
        code_assignt(q, binary_relation_exprt(cv, ">=", var_expr(n.ctr_PV))));
    else
      blk.copy_to_operands(
        code_assignt(q, binary_relation_exprt(cv, ">=", zero)));
  }
  else // CTD
  {
    symbol_exprt cd = var_expr(n.ctr_CD);
    symbol_exprt cv = var_expr(n.ctr_CV);
    symbol_exprt q = var_expr(n.ctr_Q);
    exprt neg_one = from_integer(BigInt(-1), int32_t_());
    symbol_exprt cd_prev =
      declare_bool_shadow(ld_name("__ctr_prev_" + n.ctr_instance));

    code_ifthenelset cd_step;
    cd_step.cond() = and_exprt(cd, not_exprt(cd_prev));
    cd_step.then_case() = code_assignt(cv, plus_exprt(cv, neg_one));
    blk.copy_to_operands(cd_step);

    blk.copy_to_operands(code_assignt(cd_prev, cd));
    blk.copy_to_operands(
      code_assignt(q, binary_relation_exprt(cv, "<=", zero)));
  }
  return blk;
}

codet ld_converter::translate_arith(const LdIRNode &n)
{
  symbol_exprt in1 = var_expr(n.arith_IN1);
  symbol_exprt out = var_expr(n.arith_OUT);

  exprt op_expr;
  switch (n.arith_kind)
  {
  case FBKind::ADD:
    op_expr = plus_exprt(in1, var_expr(n.arith_IN2));
    break;
  case FBKind::SUB:
    op_expr = exprt(exprt::minus, int32_t_());
    op_expr.copy_to_operands(in1, var_expr(n.arith_IN2));
    break;
  case FBKind::MUL:
    op_expr = mult_exprt(in1, var_expr(n.arith_IN2));
    break;
  case FBKind::DIV:
    op_expr = exprt(exprt::div, int32_t_());
    op_expr.copy_to_operands(in1, var_expr(n.arith_IN2));
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
    exprt rung_pf = true_exprt();

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
// Function emission
// -----------------------------------------------------------------------

// ld::scan_loop contains the infinite scan loop so __ESBMC_main stays loop-free
// (k-induction, non-termination, and termination passes assume __ESBMC_main is
// loop-free).
void ld_converter::emit_scan_function(const code_blockt &scan_body)
{
  code_typet scan_type;
  scan_type.return_type() = empty_typet();

  symbolt scan_sym;
  scan_sym.id = "ld::scan_loop";
  scan_sym.name = "scan_loop";
  scan_sym.module = "ld";
  scan_sym.mode = "LD";
  scan_sym.set_type(scan_type);
  scan_sym.lvalue = true;
  scan_sym.is_extern = false;
  scan_sym.file_local = false;
  scan_sym.static_lifetime = false;
  locationt loc;
  loc.set_file(ir_.source_file);
  scan_sym.location = loc;

  code_whilet loop;
  loop.cond() = true_exprt();
  loop.body() = scan_body;

  code_blockt body;
  body.copy_to_operands(loop);
  scan_sym.set_value(body);
  context_.move_symbol_to_context(scan_sym);
}

// __ESBMC_main: static init + call to ld::scan_loop(); loop-free.
void ld_converter::emit_main_function()
{
  code_typet main_type;
  main_type.return_type() = empty_typet();

  symbolt main_sym;
  main_sym.id = "__ESBMC_main";
  main_sym.name = "__ESBMC_main";
  main_sym.module = "ld";
  main_sym.mode = "LD";
  main_sym.set_type(main_type);
  main_sym.lvalue = true;
  main_sym.is_extern = false;
  main_sym.file_local = false;
  main_sym.static_lifetime = false;
  locationt loc;
  loc.set_file(ir_.source_file);
  main_sym.location = loc;

  code_blockt main_body;

  code_typet scan_type;
  scan_type.return_type() = empty_typet();
  code_function_callt call;
  call.function() = symbol_exprt("ld::scan_loop", scan_type);
  main_body.copy_to_operands(call);

  main_sym.set_value(main_body);
  context_.move_symbol_to_context(main_sym);
}

// Prepend static init assignments to __ESBMC_main.
// Called after all symbols (including property_encoder's) are in the context.
void ld_converter::prepend_static_init()
{
  symbolt *main_sym = context_.find_symbol("__ESBMC_main");
  if (!main_sym)
    return;

  code_blockt init_block;
  context_.foreach_operand_in_order([&init_block](const symbolt &s) {
    if (s.static_lifetime && !s.get_value().is_nil() && !s.get_type().is_code())
    {
      code_assignt assign(symbol_expr(s), s.get_value());
      assign.location() = s.location;
      init_block.copy_to_operands(assign);
    }
  });

  exprt old_body = main_sym->get_value();
  for (const auto &op : old_body.operands())
    init_block.copy_to_operands(op);
  main_sym->set_value(init_block);
}

// -----------------------------------------------------------------------
// Top-level convert()
// -----------------------------------------------------------------------

void ld_converter::convert()
{
  for (const auto &v : ir_.variables)
    declare_variable(v);

  code_blockt scan_body = build_scan_body(true_exprt());
  emit_scan_function(scan_body);
  emit_main_function();
}
