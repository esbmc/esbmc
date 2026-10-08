#include <ld-frontend/semantics/type_checker.h>
#include <algorithm>
#include <cstdint>
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
  auto it = var_types_.find(var);
  if (it == var_types_.end())
    throw TypeCheckError(loc_str(loc) + ": undeclared variable '" + var + "'");
  return it->second;
}

// The converter updates a block's outputs while its inputs are still read
// (a timer writes Q before it records IN, writes ET before it compares PT, a
// counter writes CV before it compares PV), so those pairs must be two
// variables.
void TypeChecker::require_distinct(
  const std::string &instance,
  const char *ports,
  const std::string &a,
  const std::string &b,
  const LdLocation &loc) const
{
  if (!a.empty() && a == b)
    throw TypeCheckError(
      loc_str(loc) + ": '" + instance + "' " + ports +
      " ports must be distinct variables");
}

void TypeChecker::require_port(
  const std::string &instance,
  const char *port,
  const std::string &var,
  const LdLocation &loc) const
{
  if (var.empty())
    throw TypeCheckError(
      loc_str(loc) + ": '" + instance + "' missing required port " + port);
}

// INT is 16 bits; DINT and TIME (a tick count) are 32.
static bool init_value_fits(const VarDecl &v)
{
  switch (v.kind)
  {
  case VarKind::INT:
    return v.init_value >= -32768 && v.init_value <= 32767;
  case VarKind::DINT:
  case VarKind::TIME:
    return v.init_value >= INT32_MIN && v.init_value <= INT32_MAX;
  default:
    return true;
  }
}

void TypeChecker::build_var_type_map(const LdAst &ast)
{
  for (const auto &v : ast.variables)
  {
    if (!init_value_fits(v))
      throw TypeCheckError(
        loc_str(v.loc) + ": initial value " + std::to_string(v.init_value) +
        " of '" + v.name + "' is outside " +
        (v.kind == VarKind::INT    ? "INT"
         : v.kind == VarKind::DINT ? "DINT"
                                   : "TIME"));
    var_types_[v.name] = v.kind;
    if (v.is_input)
      inputs_.insert(v.name);
  }
}

// -----------------------------------------------------------------------
// Per-element checks
// -----------------------------------------------------------------------

void TypeChecker::check_timer_fb(const TimerFBNode &fb)
{
  require_port(fb.instance_name, "IN", fb.IN_var, fb.loc);
  require_port(fb.instance_name, "PT", fb.PT_var, fb.loc);
  require_port(fb.instance_name, "Q", fb.Q_var, fb.loc);
  require_port(fb.instance_name, "ET", fb.ET_var, fb.loc);

  if (lookup_type(fb.IN_var, fb.loc) != VarKind::BOOL)
    throw TypeCheckError(
      loc_str(fb.loc) + ": timer '" + fb.instance_name +
      "' IN port requires BOOL");

  auto pt_kind = lookup_type(fb.PT_var, fb.loc);
  if (
    pt_kind != VarKind::TIME && pt_kind != VarKind::INT &&
    pt_kind != VarKind::DINT)
    throw TypeCheckError(
      loc_str(fb.loc) + ": timer '" + fb.instance_name +
      "' PT port requires TIME/INT/DINT");

  if (lookup_type(fb.Q_var, fb.loc) != VarKind::BOOL)
    throw TypeCheckError(
      loc_str(fb.loc) + ": timer '" + fb.instance_name +
      "' Q port requires BOOL");

  auto et_kind = lookup_type(fb.ET_var, fb.loc);
  if (
    et_kind != VarKind::TIME && et_kind != VarKind::INT &&
    et_kind != VarKind::DINT)
    throw TypeCheckError(
      loc_str(fb.loc) + ": timer '" + fb.instance_name +
      "' ET port requires TIME/INT/DINT");

  if (et_kind == VarKind::INT && pt_kind != VarKind::INT)
    throw TypeCheckError(
      loc_str(fb.loc) + ": timer '" + fb.instance_name +
      "' ET port is narrower than PT");

  require_distinct(fb.instance_name, "IN and Q", fb.IN_var, fb.Q_var, fb.loc);
  require_distinct(fb.instance_name, "PT and ET", fb.PT_var, fb.ET_var, fb.loc);
}

void TypeChecker::check_counter_fb(const CounterFBNode &fb)
{
  // CU (CTU) / CD (CTD) is the edge trigger — required.
  // CV and Q are required; PV and R are optional.
  if (fb.kind == FBKind::CTU)
    require_port(fb.instance_name, "CU", fb.CU_var, fb.loc);
  else
    require_port(fb.instance_name, "CD", fb.CD_var, fb.loc);

  require_port(fb.instance_name, "CV", fb.CV_var, fb.loc);
  require_port(fb.instance_name, "Q", fb.Q_var, fb.loc);

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
  check_bool(fb.LD_var, "LD");
  check_bool(fb.Q_var, "Q");
  check_int(fb.PV_var, "PV");
  check_int(fb.CV_var, "CV");

  if (
    !fb.PV_var.empty() && lookup_type(fb.CV_var, fb.loc) == VarKind::INT &&
    lookup_type(fb.PV_var, fb.loc) != VarKind::INT)
    throw TypeCheckError(
      loc_str(fb.loc) + ": counter '" + fb.instance_name +
      "' CV port is narrower than PV");
  require_distinct(fb.instance_name, "PV and CV", fb.PV_var, fb.CV_var, fb.loc);
}

void TypeChecker::check_arith_fb(const ArithFBNode &fb)
{
  require_port(fb.instance_name, "IN1", fb.IN1_var, fb.loc);
  require_port(fb.instance_name, "OUT", fb.OUT_var, fb.loc);
  if (fb.kind != FBKind::MOVE)
    require_port(fb.instance_name, "IN2", fb.IN2_var, fb.loc);

  auto check_numeric = [&](const std::string &var, const char *port) {
    auto k = lookup_type(var, fb.loc);
    if (k != VarKind::INT && k != VarKind::DINT)
      throw TypeCheckError(
        loc_str(fb.loc) + ": arith FB '" + fb.instance_name + "' " + port +
        " port requires INT/DINT");
  };
  check_numeric(fb.IN1_var, "IN1");
  check_numeric(fb.OUT_var, "OUT");
  if (fb.kind != FBKind::MOVE)
    check_numeric(fb.IN2_var, "IN2");
}

void TypeChecker::check_rung_element(const RungElement &elem)
{
  switch (elem.kind)
  {
  case RungElementKind::Contact:
    if (elem.contact.variable.empty())
      throw TypeCheckError(loc_str(elem.loc) + ": contact has no variable");
    // Non-BOOL contacts/coils occur in analog process control (INT/REAL
    // process variables). They are permitted; the converter coerces a numeric
    // contact to a Boolean test (var != 0) and casts power-flow to a numeric
    // coil. Only the variable's existence is enforced here (lookup_type throws
    // on an undeclared name), so a stray contact cannot slip through as nondet.
    lookup_type(elem.contact.variable, elem.loc);
    break;

  case RungElementKind::Coil:
    if (elem.coil.variable.empty())
      throw TypeCheckError(loc_str(elem.loc) + ": coil has no variable");
    lookup_type(elem.coil.variable, elem.loc);
    if (inputs_.count(elem.coil.variable))
      throw TypeCheckError(
        loc_str(elem.loc) + ": coil on input '" + elem.coil.variable + "'");
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

  // Second, independent guard on the vacuity of #7354: the parser's body
  // whitelist is keyed on notation, this catches an empty model however it
  // arose (empty <LD/>, namespace-prefixed document, a dropped element).
  const bool has_logic =
    std::any_of(
      ast.networks.begin(),
      ast.networks.end(),
      [](const NetworkNode &n) { return !n.rungs.empty(); }) ||
    !ast.user_fb_instances.empty();
  if (!has_logic)
    throw TypeCheckError(
      ast.source_file +
      ": no ladder logic was translated; the scan cycle "
      "would be empty and every property vacuous");

  for (const auto &net : ast.networks)
    for (const auto &rung : net.rungs)
      for (const auto &elem : rung.elements)
        check_rung_element(elem);
}
