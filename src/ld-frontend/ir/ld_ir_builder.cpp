#include <ld-frontend/ir/ld_ir_builder.h>

LdIRNode LdIRBuilder::lower_element(const RungElement &elem)
{
  LdIRNode node;
  node.loc = elem.loc;

  switch (elem.kind)
  {
  case RungElementKind::Contact:
    node.kind = LdIRNodeKind::ContactEval;
    node.variable = elem.contact.variable;
    node.contact_kind = elem.contact.kind;
    node.rule = (elem.contact.kind == ContactKind::NormallyOpen)
                  ? SosRule::NO_Contact_True
                  : SosRule::NC_Contact_True;
    break;

  case RungElementKind::Coil:
    node.kind = LdIRNodeKind::CoilAssign;
    node.variable = elem.coil.variable;
    node.coil_kind = elem.coil.kind;
    switch (elem.coil.kind)
    {
    case CoilKind::Output:
      node.rule = SosRule::Output_Coil;
      break;
    case CoilKind::Set:
      node.rule = SosRule::Set_Coil;
      break;
    case CoilKind::Reset:
      node.rule = SosRule::Reset_Coil;
      break;
    }
    break;

  case RungElementKind::TimerFB:
    node.kind = LdIRNodeKind::TimerStep;
    node.timer_kind = elem.timer_fb.kind;
    node.timer_IN = elem.timer_fb.IN_var;
    node.timer_ET = elem.timer_fb.ET_var;
    node.timer_PT = elem.timer_fb.PT_var;
    node.timer_Q = elem.timer_fb.Q_var;
    switch (elem.timer_fb.kind)
    {
    case FBKind::TON:
      node.rule = SosRule::TON_Step;
      break;
    case FBKind::TOF:
      node.rule = SosRule::TOF_Step;
      break;
    case FBKind::TP:
      node.rule = SosRule::TP_Step;
      break;
    default:
      node.rule = SosRule::TON_Step;
      break;
    }
    break;

  case RungElementKind::CounterFB:
    node.kind = LdIRNodeKind::CounterStep;
    node.ctr_kind = elem.counter_fb.kind;
    node.ctr_instance = elem.counter_fb.instance_name;
    node.ctr_CU = elem.counter_fb.CU_var;
    node.ctr_CD = elem.counter_fb.CD_var;
    node.ctr_R = elem.counter_fb.R_var;
    node.ctr_CV = elem.counter_fb.CV_var;
    node.ctr_PV = elem.counter_fb.PV_var;
    node.ctr_Q = elem.counter_fb.Q_var;
    node.rule = (elem.counter_fb.kind == FBKind::CTU) ? SosRule::CTU_Step
                                                      : SosRule::CTD_Step;
    break;

  case RungElementKind::ArithFB:
    node.kind = LdIRNodeKind::ArithStep;
    node.arith_kind = elem.arith_fb.kind;
    node.arith_IN1 = elem.arith_fb.IN1_var;
    node.arith_IN2 = elem.arith_fb.IN2_var;
    node.arith_OUT = elem.arith_fb.OUT_var;
    node.rule = SosRule::Arith_Step;
    break;
  }

  return node;
}

LdIRRung LdIRBuilder::lower_rung(const RungNode &rung)
{
  LdIRRung ir_rung;
  ir_rung.id = rung.id;
  ir_rung.loc = rung.loc;
  for (const auto &elem : rung.elements)
    ir_rung.nodes.push_back(lower_element(elem));
  return ir_rung;
}

LdIR LdIRBuilder::build(const LdAst &ast)
{
  LdIR ir;
  ir.source_file = ast.source_file;
  ir.variables = ast.variables;

  for (const auto &net : ast.networks)
    for (const auto &rung : net.rungs)
      ir.rungs.push_back(lower_rung(rung));

  // Join user-FB instances with their definitions so the converter can
  // execute each instance's translated body once per scan cycle.
  for (const auto &inst : ast.user_fb_instances)
  {
    for (const auto &def : ast.user_fb_defs)
    {
      if (def.type_name != inst.type_name)
        continue;
      UserFBExec ex;
      ex.type_name = def.type_name;
      ex.instance_name = inst.instance_name;
      ex.input_vars = def.input_vars;
      ex.local_vars = def.local_vars;
      ex.output_var = def.output_var;
      ex.output_kind = def.output_kind;
      ex.in1_var = inst.in1_var;
      ex.out_wires = inst.out_wires;
      ex.st_body = def.st_body;
      ir.user_fbs.push_back(ex);
      break;
    }
  }

  return ir;
}
