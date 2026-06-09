#include <ld-frontend/semantics/type_checker.h>
#include <sstream>

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

std::string TypeChecker::loc_str(const LdLocation &loc)
{
  std::ostringstream s;
  s << loc.file << ":" << loc.line << ":" << loc.col;
  return s.str();
}

VarKind
TypeChecker::lookup_type(const std::string &var, const LdLocation &loc) const
{
  if (var.empty())
    return VarKind::BOOL;
  auto it = var_types_.find(var);
  if (it == var_types_.end())
    throw TypeCheckError(loc_str(loc) + ": undeclared variable '" + var + "'");
  return it->second;
}

void TypeChecker::build_var_type_map(const LdAst &ast)
{
  for (const auto &v : ast.variables)
    var_types_[v.name] = v.kind;
}

// -----------------------------------------------------------------------
// Per-element checks
// -----------------------------------------------------------------------

void TypeChecker::check_timer_fb(const TimerFBNode &fb)
{
  // IN must be BOOL; PT must be TIME (tick count); Q must be BOOL; ET must be TIME.
  if (!fb.IN_var.empty())
  {
    auto kind = lookup_type(fb.IN_var, fb.loc);
    if (kind != VarKind::BOOL)
      throw TypeCheckError(
        loc_str(fb.loc) + ": timer '" + fb.instance_name +
        "' IN port requires BOOL variable, got " + fb.IN_var);
  }
  if (!fb.PT_var.empty())
  {
    auto kind = lookup_type(fb.PT_var, fb.loc);
    if (kind != VarKind::TIME && kind != VarKind::INT && kind != VarKind::DINT)
      throw TypeCheckError(
        loc_str(fb.loc) + ": timer '" + fb.instance_name +
        "' PT port requires TIME/INT variable");
  }
  if (!fb.Q_var.empty())
  {
    auto kind = lookup_type(fb.Q_var, fb.loc);
    if (kind != VarKind::BOOL)
      throw TypeCheckError(
        loc_str(fb.loc) + ": timer '" + fb.instance_name +
        "' Q port requires BOOL variable");
  }
}

void TypeChecker::check_counter_fb(const CounterFBNode &fb)
{
  // CU/CD/R/Q must be BOOL; PV/CV must be INT or DINT.
  auto check_bool = [&](const std::string &var, const char *port) {
    if (!var.empty() && lookup_type(var, fb.loc) != VarKind::BOOL)
      throw TypeCheckError(
        loc_str(fb.loc) + ": counter '" + fb.instance_name + "' " + port +
        " port requires BOOL");
  };
  auto check_int = [&](const std::string &var, const char *port) {
    if (var.empty())
      return;
    auto k = lookup_type(var, fb.loc);
    if (k != VarKind::INT && k != VarKind::DINT)
      throw TypeCheckError(
        loc_str(fb.loc) + ": counter '" + fb.instance_name + "' " + port +
        " port requires INT/DINT");
  };
  check_bool(fb.CU_var, "CU");
  check_bool(fb.CD_var, "CD");
  check_bool(fb.R_var, "R");
  check_bool(fb.Q_var, "Q");
  check_int(fb.PV_var, "PV");
  check_int(fb.CV_var, "CV");
}

void TypeChecker::check_arith_fb(const ArithFBNode &fb)
{
  // IN1/IN2/OUT must be INT or DINT.
  auto check_numeric = [&](const std::string &var, const char *port) {
    if (var.empty())
      return;
    auto k = lookup_type(var, fb.loc);
    if (k != VarKind::INT && k != VarKind::DINT)
      throw TypeCheckError(
        loc_str(fb.loc) + ": arith FB '" + fb.instance_name + "' " + port +
        " port requires INT/DINT");
  };
  check_numeric(fb.IN1_var, "IN1");
  check_numeric(fb.IN2_var, "IN2");
  check_numeric(fb.OUT_var, "OUT");
}

void TypeChecker::check_rung_element(const RungElement &elem)
{
  switch (elem.kind)
  {
  case RungElementKind::Contact:
    if (!elem.contact.variable.empty())
      if (lookup_type(elem.contact.variable, elem.loc) != VarKind::BOOL)
        throw TypeCheckError(
          loc_str(elem.loc) + ": contact variable '" + elem.contact.variable +
          "' must be BOOL");
    break;

  case RungElementKind::Coil:
    if (!elem.coil.variable.empty())
      if (lookup_type(elem.coil.variable, elem.loc) != VarKind::BOOL)
        throw TypeCheckError(
          loc_str(elem.loc) + ": coil variable '" + elem.coil.variable +
          "' must be BOOL");
    break;

  case RungElementKind::TimerFB:
    check_timer_fb(elem.timer_fb);
    break;

  case RungElementKind::CounterFB:
    check_counter_fb(elem.counter_fb);
    break;

  case RungElementKind::ArithFB:
    check_arith_fb(elem.arith_fb);
    break;
  }
}

// -----------------------------------------------------------------------
// Top-level check
// -----------------------------------------------------------------------

void TypeChecker::check(const LdAst &ast)
{
  build_var_type_map(ast);

  for (const auto &net : ast.networks)
    for (const auto &rung : net.rungs)
      for (const auto &elem : rung.elements)
        check_rung_element(elem);
}
