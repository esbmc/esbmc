#include <ld-frontend/parser/plcopen_xml_parser.h>
#include <pugixml.hpp>
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <functional>
#include <map>
#include <set>

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

static LdLocation
loc_from_node(const pugi::xml_node &n, const std::string &file)
{
  LdLocation loc;
  loc.file = file;
  // PLCopen XML does not standardise source coordinates; use the
  // lineNumber/columnNumber attributes when present.
  if (auto line_attr = n.attribute("lineNumber"))
    loc.line = static_cast<unsigned>(line_attr.as_uint());
  if (auto col_attr = n.attribute("columnNumber"))
    loc.col = static_cast<unsigned>(col_attr.as_uint());
  return loc;
}

static std::string text_or_attr(
  const pugi::xml_node &n,
  const char *attr_name,
  const char *child_name = nullptr)
{
  if (auto a = n.attribute(attr_name))
    return a.as_string();
  if (child_name)
    if (auto c = n.child(child_name))
      return c.child_value();
  return {};
}

// -----------------------------------------------------------------------
// PlcopenXmlParser
// -----------------------------------------------------------------------

VarKind PlcopenXmlParser::var_kind_from_string(const std::string &s)
{
  static const std::unordered_map<std::string, VarKind> table = {
    {"BOOL", VarKind::BOOL},
    {"INT", VarKind::INT},
    {"DINT", VarKind::DINT},
    {"UINT", VarKind::INT},
    {"SINT", VarKind::INT},
    {"LINT", VarKind::DINT},
    {"WORD", VarKind::INT},
    {"TIME", VarKind::TIME},
    {"REAL", VarKind::REAL},
    {"LREAL", VarKind::REAL},
  };
  auto it = table.find(s);
  if (it == table.end())
    return VarKind::BOOL; // default; type checker will flag unsupported types
  return it->second;
}

// Accepts both the element name (NormallyClosedContact) and the value of a
// negated="..." attribute. Vendors spell the attribute "true"/"false" or
// repeat the attribute name, so "false" must not be read as negation.
ContactKind PlcopenXmlParser::contact_kind_from_string(const std::string &s)
{
  if (s == "negated" || s == "true" || s == "NormallyClosedContact")
    return ContactKind::NormallyClosed;
  return ContactKind::NormallyOpen;
}

// PLCopen writes the transition-sensing kind as edge="rising|falling"; some
// vendors emit the IEC operator letter (P/N) or "positive"/"negative".
static ContactEdge contact_edge_from_string(const std::string &s)
{
  if (s == "rising" || s == "positive" || s == "R" || s == "P")
    return ContactEdge::Rising;
  if (s == "falling" || s == "negative" || s == "F" || s == "N")
    return ContactEdge::Falling;
  return ContactEdge::None;
}

CoilKind PlcopenXmlParser::coil_kind_from_string(const std::string &s)
{
  if (s == "set" || s == "SetCoil")
    return CoilKind::Set;
  if (s == "reset" || s == "ResetCoil")
    return CoilKind::Reset;
  return CoilKind::Output;
}

static FBKind fb_kind_of(const std::string &s);
static bool
literal_to_ticks(const std::string &text, unsigned interval_ms, long long &out);

FBKind PlcopenXmlParser::fb_kind_from_string(const std::string &s)
{
  return fb_kind_of(s);
}

// -----------------------------------------------------------------------
// Variable declaration parsing
// -----------------------------------------------------------------------

VarDecl PlcopenXmlParser::parse_var_decl(const void *node_ptr)
{
  const auto &n = *static_cast<const pugi::xml_node *>(node_ptr);
  VarDecl v;
  v.name = n.attribute("name").as_string();
  // <type><BOOL/>, <INT/>, etc. or <type><derived name="MyType"/>.
  auto type_node = n.child("type");
  std::string type_str;
  if (auto first = type_node.first_child(); !first.empty())
  {
    std::string tag = first.name();
    type_str = (tag == "derived") ? first.attribute("name").as_string() : tag;
  }
  if (type_str.empty())
    type_str = "BOOL";
  v.kind = var_kind_from_string(type_str);
  v.loc = loc_from_node(n, source_file_);

  // <initialValue><simpleValue value="2"/></initialValue>. Without this a
  // declared timer preset reads as zero, which makes TON fire immediately and
  // TOF never hold — a silently wrong model rather than a diagnosable one.
  if (auto init = n.child("initialValue").child("simpleValue"))
  {
    const std::string text = init.attribute("value").as_string("");
    long long value = 0;
    if (text == "TRUE" || text == "true")
      v.init_value = 1;
    else if (text == "FALSE" || text == "false")
      v.init_value = 0;
    else if (literal_to_ticks(text, scan_interval_ms_, value))
      v.init_value = value;
    else if (!text.empty())
      std::cerr << "warning: LD: variable '" << v.name
                << "' has an unrecognised initial value '" << text
                << "'; using 0.\n";
  }

  // OpenPLC / CONTROLLINO export all variables as <localVars> with hardware
  // addresses: %IX... = physical input, %QX... = physical output.
  // Use the address attribute to set is_input / is_output when the parent
  // tag does not already encode direction (inputVars / outputVars).
  std::string addr = n.attribute("address").as_string("");
  if (!addr.empty())
  {
    if (addr.rfind("%I", 0) == 0 || addr.rfind("%i", 0) == 0)
      v.is_input = true;
    else if (addr.rfind("%Q", 0) == 0 || addr.rfind("%q", 0) == 0)
      v.is_output = true;
  }

  return v;
}

// -----------------------------------------------------------------------
// Rung element parsing
// -----------------------------------------------------------------------

RungElement PlcopenXmlParser::parse_rung_element(const void *node_ptr)
{
  const auto &n = *static_cast<const pugi::xml_node *>(node_ptr);
  const std::string tag = n.name();
  RungElement elem;
  elem.loc = loc_from_node(n, source_file_);

  if (
    tag == "contact" || tag == "Contact" || tag == "NormallyOpenContact" ||
    tag == "NormallyClosedContact")
  {
    elem.kind = RungElementKind::Contact;
    std::string neg = text_or_attr(n, "negated", nullptr);
    elem.contact.kind = contact_kind_from_string(neg.empty() ? tag : neg);
    elem.contact.edge =
      contact_edge_from_string(text_or_attr(n, "edge", nullptr));
    // Variable connected to the contact
    if (auto var_node = n.child("variable"))
      elem.contact.variable = var_node.child_value();
    else
      elem.contact.variable = text_or_attr(n, "variable", "variable");
    elem.contact.loc = elem.loc;
    return elem;
  }

  if (tag == "coil" || tag == "Coil" || tag == "SetCoil" || tag == "ResetCoil")
  {
    elem.kind = RungElementKind::Coil;
    // PLCopen XML encodes coil kind either as the tag name (SetCoil/ResetCoil)
    // or as a "kind" attribute on a generic <coil kind="set|reset"/> element.
    std::string kind_str = text_or_attr(n, "kind", nullptr);
    elem.coil.kind = kind_str.empty() ? coil_kind_from_string(tag)
                                      : coil_kind_from_string(kind_str);
    if (auto var_node = n.child("variable"))
      elem.coil.variable = var_node.child_value();
    else
      elem.coil.variable = text_or_attr(n, "variable", "variable");
    elem.coil.loc = elem.loc;
    return elem;
  }

  if (tag == "block" || tag == "Block")
  {
    const std::string fb_type = text_or_attr(n, "typeName", "typeName");
    FBKind kind = fb_kind_from_string(fb_type);
    const std::string inst = text_or_attr(n, "instanceName", "instanceName");

    auto get_var = [&](const char *port) -> std::string {
      for (auto var : n.children("variable"))
        if (std::string(var.attribute("formalParameter").as_string()) == port)
          return var.child_value();
      return {};
    };

    if (kind == FBKind::TON || kind == FBKind::TOF || kind == FBKind::TP)
    {
      elem.kind = RungElementKind::TimerFB;
      elem.timer_fb.kind = kind;
      elem.timer_fb.instance_name = inst;
      elem.timer_fb.IN_var = get_var("IN");
      elem.timer_fb.PT_var = get_var("PT");
      elem.timer_fb.Q_var = get_var("Q");
      elem.timer_fb.ET_var = get_var("ET");
      elem.timer_fb.loc = elem.loc;
      return elem;
    }

    if (kind == FBKind::CTU || kind == FBKind::CTD)
    {
      elem.kind = RungElementKind::CounterFB;
      elem.counter_fb.kind = kind;
      elem.counter_fb.instance_name = inst;
      elem.counter_fb.CU_var = get_var("CU");
      elem.counter_fb.CD_var = get_var("CD");
      elem.counter_fb.R_var = get_var("R");
      elem.counter_fb.PV_var = get_var("PV");
      elem.counter_fb.Q_var = get_var("Q");
      elem.counter_fb.CV_var = get_var("CV");
      elem.counter_fb.loc = elem.loc;
      return elem;
    }

    // Arithmetic FB
    elem.kind = RungElementKind::ArithFB;
    elem.arith_fb.kind = kind;
    elem.arith_fb.instance_name = inst;
    elem.arith_fb.IN1_var = get_var("IN1");
    elem.arith_fb.IN2_var = get_var("IN2");
    elem.arith_fb.OUT_var = get_var("OUT");
    elem.arith_fb.loc = elem.loc;
    return elem;
  }

  throw UnsupportedConstructError(tag, 2);
}

// -----------------------------------------------------------------------
// Rung / network parsing
// -----------------------------------------------------------------------

RungNode PlcopenXmlParser::parse_rung(const void *node_ptr)
{
  const auto &n = *static_cast<const pugi::xml_node *>(node_ptr);
  RungNode rung;
  rung.id = text_or_attr(n, "localId", "localId");
  rung.loc = loc_from_node(n, source_file_);

  for (auto child : n.children())
  {
    if (child.type() != pugi::node_element)
      continue;
    rung.elements.push_back(parse_rung_element(&child));
  }
  return rung;
}
// -----------------------------------------------------------------------
// Graphical LD (tc6_0201) resolver
// -----------------------------------------------------------------------
// In graphical PLCopen XML the ladder logic is encoded as a connection
// graph: each element carries a localId and its inputs are listed as
// <connection refLocalId="..."/> children.  The textual <rung> wrapper is
// absent.  The graph is resolved by enumerating every simple path from a
// leftPowerRail to each sink (a coil, or the enable pin of a function
// block); each path is a series contact chain (AND) and the paths reaching
// one sink are alternatives (OR).
//
// Rungs are emitted per sink in rightPowerRail order, which is the order in
// which the vendor tool draws them and therefore the scan execution order.
// A function block encountered on a path is emitted just before the first
// sink that consumes it, so a block still observes the values written by the
// rungs drawn above it.
//
// Returns true if the LD body contained graphical elements and rungs
// were successfully extracted; false if this is a textual LD body.

struct GNode
{
  std::string tag;      // "contact", "coil", "leftPowerRail", "block", ...
  std::string var;      // variable name (contacts and coils)
  bool negated = false; // normally-closed contact
  ContactEdge edge = ContactEdge::None;
  std::string storage;                // "set", "reset", or "" for normal coil
  std::string type_name;              // block typeName (TON, CTU, ...)
  std::string instance_name;          // block instanceName
  std::string expression;             // inVariable literal text (T#20s, 5, ...)
  std::map<std::string, int> in_pins; // formalParameter -> source localId
  std::vector<int> feeds; // forward edges (this node feeds these localIds)
};

// Parse an IEC 61131-3 duration literal (T#20s, TIME#1m30s, t#500ms) into
// milliseconds.  Returns -1 when the text is not a duration literal.
static long long parse_duration_ms(const std::string &text)
{
  std::string s;
  for (char c : text)
    if (!isspace(static_cast<unsigned char>(c)))
      s += static_cast<char>(toupper(static_cast<unsigned char>(c)));

  size_t hash = s.find('#');
  if (hash == std::string::npos)
    return -1;
  std::string prefix = s.substr(0, hash);
  if (prefix != "T" && prefix != "TIME")
    return -1;

  const std::string body = s.substr(hash + 1);
  long long total = 0;
  size_t i = 0;
  bool any = false;
  while (i < body.size())
  {
    size_t num_start = i;
    while (i < body.size() && (isdigit(static_cast<unsigned char>(body[i])) ||
                               body[i] == '.' || body[i] == '_'))
      ++i;
    if (i == num_start)
      return -1;
    std::string num_text;
    for (size_t k = num_start; k < i; ++k)
      if (body[k] != '_')
        num_text += body[k];

    size_t unit_start = i;
    while (i < body.size() && isalpha(static_cast<unsigned char>(body[i])))
      ++i;
    const std::string unit = body.substr(unit_start, i - unit_start);

    double scale = 0;
    if (unit == "MS")
      scale = 1;
    else if (unit == "S")
      scale = 1000;
    else if (unit == "M")
      scale = 60.0 * 1000;
    else if (unit == "H")
      scale = 3600.0 * 1000;
    else if (unit == "D")
      scale = 24.0 * 3600.0 * 1000;
    else
      return -1;

    total += static_cast<long long>(atof(num_text.c_str()) * scale);
    any = true;
  }
  return any ? total : -1;
}

// Resolve an <inVariable> literal to the value the fixed-tick model expects.
// A duration is converted to scan ticks using the configured task interval
// (§3.3: one scan iteration advances time by exactly one tick); anything else
// is read as a plain integer.  Returns false when the text is neither.
static bool
literal_to_ticks(const std::string &text, unsigned interval_ms, long long &out)
{
  const long long ms = parse_duration_ms(text);
  if (ms >= 0)
  {
    const long long period = interval_ms ? interval_ms : 1;
    // Round up: a preset shorter than one scan still takes one scan to expire.
    out = (ms + period - 1) / period;
    return true;
  }

  // Validate rather than convert-and-catch. A non-numeric preset — a data pin
  // wired to a named variable, or an unparsable <initialValue> — used to
  // terminate the process on std::stoll's std::invalid_argument despite the
  // catch that was here, leaving both callers' fallback paths unreachable.
  // Parsers should not route ordinary "not a number" input through exceptions.
  errno = 0;
  char *end = nullptr;
  const long long v = std::strtoll(text.c_str(), &end, 10);
  if (end == text.c_str() || errno == ERANGE)
    return false;
  out = v;
  return true;
}

// The FB type table is shared between the textual rung parser and the
// graphical resolver, which is a free function and cannot reach the member.
static FBKind fb_kind_of(const std::string &s)
{
  static const std::unordered_map<std::string, FBKind> table = {
    {"TON", FBKind::TON},
    {"TOF", FBKind::TOF},
    {"TP", FBKind::TP},
    {"CTU", FBKind::CTU},
    {"CTD", FBKind::CTD},
    {"ADD", FBKind::ADD},
    {"SUB", FBKind::SUB},
    {"MUL", FBKind::MUL},
    {"DIV", FBKind::DIV},
    {"MOVE", FBKind::MOVE},
  };
  auto it = table.find(s);
  if (it == table.end())
    throw LdParseError("Unknown FB type: " + s);
  return it->second;
}

static bool is_coil_tag(const std::string &t)
{
  return t == "coil" || t == "SetCoil" || t == "ResetCoil";
}

static bool parse_graphical_ld(
  const pugi::xml_node &ld_body,
  NetworkNode &net,
  const std::string &source_file,
  std::vector<VarDecl> &synth_vars,
  unsigned interval_ms)
{
  // Step 1: collect all top-level elements and detect graphical format.
  // A graphical LD body has <contact>, <coil>, <leftPowerRail> as direct
  // children; a textual LD body has <rung> children.
  bool has_rung = false;
  bool has_graphical = false;
  for (auto child : ld_body.children())
  {
    std::string t = child.name();
    if (t == "rung" || t == "Rung")
      has_rung = true;
    if (t == "leftPowerRail" || t == "contact" || t == "coil")
      has_graphical = true;
  }
  if (has_rung || !has_graphical)
    return false; // textual or empty — handled by existing parse_rung path

  // Step 2: build node table indexed by localId.
  std::unordered_map<int, GNode> nodes;
  for (auto child : ld_body.children())
  {
    std::string t = child.name();
    int lid = child.attribute("localId").as_int(-1);
    if (lid < 0)
      continue;

    GNode g;
    g.tag = t;
    if (auto v = child.child("variable"))
      g.var = v.child_value();
    std::string neg_attr = child.attribute("negated").as_string("");
    g.negated = (neg_attr == "true" || neg_attr == "negated");
    g.edge = contact_edge_from_string(child.attribute("edge").as_string(""));

    std::string storage_attr = child.attribute("storage").as_string("");
    if (storage_attr.empty())
    {
      if (t == "SetCoil")
        storage_attr = "set";
      if (t == "ResetCoil")
        storage_attr = "reset";
    }
    g.storage = storage_attr;

    if (t == "block" || t == "Block")
    {
      g.type_name = child.attribute("typeName").as_string("");
      g.instance_name = child.attribute("instanceName").as_string("");
      // Record each input pin's source so data pins (PT, PV) can be resolved
      // without turning them into power-flow edges.
      for (auto pin : child.child("inputVariables").children("variable"))
      {
        const std::string formal =
          pin.attribute("formalParameter").as_string("");
        auto conn = pin.select_node(".//connection").node();
        const int src = conn.attribute("refLocalId").as_int(-1);
        if (!formal.empty() && src >= 0)
          g.in_pins[formal] = src;
      }
    }

    if (t == "inVariable")
      g.expression = child.child_value("expression");

    nodes[lid] = g;
  }

  // Step 3: build forward edges from backward (refLocalId) edges.  Only
  // power-flow connections become edges: a block's data pins (PT, PV) feed
  // values, not power, and must not create rung paths through the block.
  static const std::set<std::string> enable_pins = {"IN", "CU", "CD"};
  for (auto child : ld_body.children())
  {
    int lid = child.attribute("localId").as_int(-1);
    if (lid < 0 || nodes.find(lid) == nodes.end())
      continue;

    if (nodes[lid].tag == "block" || nodes[lid].tag == "Block")
    {
      for (const auto &[formal, src] : nodes[lid].in_pins)
        if (enable_pins.count(formal) && nodes.count(src))
          nodes[src].feeds.push_back(lid);
      continue;
    }

    for (auto conn_node : child.select_nodes(".//connection"))
    {
      auto conn = conn_node.node();
      int src = conn.attribute("refLocalId").as_int(-1);
      if (src >= 0 && nodes.count(src))
        nodes[src].feeds.push_back(lid);
    }
  }

  // Step 4: invert the power-flow edges and mark which nodes the rails reach.
  //
  // Power flow is computed per node, not per rail-to-sink path:
  //
  //     pf(n) = (OR over predecessors p of pf(p)) AND cond(n)
  //
  // which needs each node's predecessors and costs O(V+E). Enumerating simple
  // paths instead — the distributed form of the same expression — is
  // exponential in the number of parallel branches, and reached 112s of
  // GOTO-creation time on a network of 36 contacts.
  //
  // Reachability is tracked separately because a node whose predecessors never
  // reach a rail has pf = false in every scan. That is indistinguishable from a
  // correctly de-energised node in the emitted IR, so such a sink is diagnosed
  // (below) rather than left to verify over strictly less behaviour.
  std::vector<int> left_rails;
  for (auto &[lid, g] : nodes)
    if (g.tag == "leftPowerRail")
      left_rails.push_back(lid);

  std::map<int, std::vector<int>> preds;
  for (auto &[lid, g] : nodes)
    for (int succ : g.feeds)
      preds[succ].push_back(lid);

  std::set<int> rail_reaches;
  {
    std::vector<int> queue(left_rails.begin(), left_rails.end());
    for (int rail : left_rails)
      rail_reaches.insert(rail);
    while (!queue.empty())
    {
      const int cur = queue.back();
      queue.pop_back();
      for (int succ : nodes.at(cur).feeds)
        if (rail_reaches.insert(succ).second)
          queue.push_back(succ);
    }
  }

  // Predecessors that carry power: a node fed only by unreachable nodes is
  // dead, and the callers treat it as an undriven sink.
  auto live_preds = [&](int lid) {
    std::vector<int> live;
    auto it = preds.find(lid);
    if (it != preds.end())
      for (int p : it->second)
        if (rail_reaches.count(p))
          live.push_back(p);
    return live;
  };

  // Step 5: rung construction helpers.
  int rung_counter = 0;
  int acc_counter = 0;
  const LdLocation loc{source_file, 0, 0};

  auto new_rung = [&]() {
    RungNode r;
    r.id = "g" + std::to_string(rung_counter++);
    r.loc = loc;
    return r;
  };

  auto make_contact =
    [&](const std::string &var, bool negated, ContactEdge edge) {
      RungElement e;
      e.loc = loc;
      e.kind = RungElementKind::Contact;
      e.contact.kind =
        negated ? ContactKind::NormallyClosed : ContactKind::NormallyOpen;
      e.contact.edge = edge;
      e.contact.variable = var;
      e.contact.loc = loc;
      return e;
    };

  auto make_coil = [&](const std::string &var, CoilKind kind) {
    RungElement e;
    e.loc = loc;
    e.kind = RungElementKind::Coil;
    e.coil.kind = kind;
    e.coil.variable = var;
    e.coil.loc = loc;
    return e;
  };

  // A variable both written by a coil and read by a contact closes a feedback
  // loop across the network. IEC 61131-3 §4.1.3 requires the loop variable to
  // be read at its value on entry to the network, so contacts read a snapshot
  // taken before any rung runs rather than whatever an earlier coil left.
  // Without this the network's meaning depends on the order the resolver
  // happens to emit its sinks in — exactly the order-dependence §3.2 rejects.
  std::set<std::string> feedback_vars;
  {
    std::set<std::string> written, sensed;
    for (auto &[lid, g] : nodes)
    {
      (void)lid;
      if (is_coil_tag(g.tag) && !g.var.empty())
        written.insert(g.var);
      if (g.tag == "contact" && !g.var.empty())
        sensed.insert(g.var);
    }
    for (const auto &v : written)
      if (sensed.count(v))
        feedback_vars.insert(v);
  }
  auto sensed_name = [&](const std::string &var) {
    return feedback_vars.count(var) ? var + "__prev" : var;
  };

  std::set<std::string> declared_synth;
  auto synth_var =
    [&](const std::string &name, VarKind kind, bool driven, long long init) {
      if (!declared_synth.insert(name).second)
        return name;
      VarDecl v;
      v.name = name;
      v.kind = kind;
      // Pins written by an FB step must not be mistaken for physical inputs by
      // the I/O inference heuristic, which would havoc them every scan.
      v.is_output = driven;
      v.init_value = init;
      v.loc = loc;
      synth_vars.push_back(v);
      return name;
    };

  auto inst_name = [&](int block_id) {
    const GNode &g = nodes.at(block_id);
    return g.instance_name.empty() ? "blk" + std::to_string(block_id)
                                   : g.instance_name;
  };

  auto pin_name = [&](int block_id, const char *pin) {
    return inst_name(block_id) + "__" + pin;
  };

  // Resolve a block data pin (PT, PV, R) to a variable name: the declared
  // variable it is wired to, or a synthesised constant holding its literal.
  auto resolve_data_pin =
    [&](int block_id, const char *pin, VarKind kind) -> std::string {
    const GNode &g = nodes.at(block_id);
    auto it = g.in_pins.find(pin);
    if (it == g.in_pins.end() || !nodes.count(it->second))
      return synth_var(pin_name(block_id, pin), kind, false, 0);

    const GNode &src = nodes.at(it->second);
    if (!src.var.empty())
      return src.var; // wired to a declared variable

    long long value = 0;
    if (
      !src.expression.empty() &&
      !literal_to_ticks(src.expression, interval_ms, value))
    {
      // An <inVariable> may hold a symbol rather than a literal; treat a bare
      // identifier as a variable reference before falling back to a constant.
      const bool identifier =
        !src.expression.empty() &&
        (isalpha(static_cast<unsigned char>(src.expression[0])) ||
         src.expression[0] == '_');
      if (identifier)
        return src.expression;
      std::cerr << "warning: graphical LD: block pin " << pin
                << " has unrecognised literal '" << src.expression
                << "'; using 0.\n";
    }
    return synth_var(pin_name(block_id, pin), kind, false, value);
  };

  // The element a node contributes to a rung that reads its power flow: a
  // contact tests its own operand, a block hands on its Q pin, and a rail is
  // unconditionally live so it contributes nothing.
  std::function<void(int)> ensure_pf;

  // Name the power flow out of a node. Contacts get one variable per node
  // rather than per operand: the same operand may appear on several rungs with
  // different upstream conditions. A block's power flow is its Q pin, already
  // assigned by its own step.
  auto pf_name = [&](int lid) {
    const GNode &g = nodes.at(lid);
    if (g.tag == "block" || g.tag == "Block")
      return pin_name(lid, "Q");
    return "pf" + std::to_string(lid);
  };

  // The element a node contributes to a rung that reads its power flow.
  auto pf_contact = [&](int lid) {
    const GNode &g = nodes.at(lid);
    if (g.tag != "contact" && g.tag != "block" && g.tag != "Block")
      throw UnsupportedConstructError(
        g.tag + (g.var.empty() ? "" : " (var=" + g.var + ")"), 2);
    return make_contact(pf_name(lid), false, ContactEdge::None);
  };

  // Emit the rungs that assign a node's power flow, once per node. The
  // accumulator is cleared and then set from each live predecessor, so the
  // whole network costs one clear plus one rung per edge.
  std::set<int> pf_emitted;
  std::set<int> pf_in_progress;
  auto emit_pf = [&](int lid) {
    const GNode &g = nodes.at(lid);
    const std::string acc = synth_var(pf_name(lid), VarKind::BOOL, true, 0);
    const auto live = live_preds(lid);

    RungNode clear = new_rung();
    clear.elements.push_back(make_coil(acc, CoilKind::Reset));
    net.rungs.push_back(std::move(clear));

    for (int p : live)
    {
      RungNode r = new_rung();
      // The rail is always live, so a node wired straight to it needs only its
      // own condition.
      if (nodes.at(p).tag != "leftPowerRail")
        r.elements.push_back(pf_contact(p));
      r.elements.push_back(make_contact(sensed_name(g.var), g.negated, g.edge));
      r.elements.push_back(make_coil(acc, CoilKind::Set));
      net.rungs.push_back(std::move(r));
    }
  };

  std::function<void(int)> emit_block;

  // Assign a node's power flow, after its predecessors'. Recursion depth is
  // bounded by the longest chain, and the in-progress set turns a cyclic
  // network into a diagnostic rather than a stack overflow.
  ensure_pf = [&](int lid) {
    const GNode &g = nodes.at(lid);
    if (g.tag == "leftPowerRail" || !pf_emitted.insert(lid).second)
      return;
    if (!pf_in_progress.insert(lid).second)
      throw LdParseError(
        "graphical LD: power flow into localId " + std::to_string(lid) +
        " is cyclic");

    for (int p : live_preds(lid))
      ensure_pf(p);

    if (g.tag == "block" || g.tag == "Block")
      emit_block(lid);
    else if (g.tag == "contact")
      emit_pf(lid);
    else if (!is_coil_tag(g.tag))
      throw UnsupportedConstructError(
        g.tag + (g.var.empty() ? "" : " (var=" + g.var + ")"), 2);

    pf_in_progress.erase(lid);
  };

  // Drive a sink from the OR of its live predecessors' power flow. A single
  // predecessor needs no accumulator and drives the sink directly.
  auto emit_sink = [&](int sink_id, const std::string &target, CoilKind kind) {
    const auto live = live_preds(sink_id);
    // Nothing would ever assign the sink, so it would hold its initial value
    // for every scan and any property over it would pass vacuously. A block
    // driving the sink whose enable pin is not one of enable_pins (step 3)
    // gets no incoming edge and lands here, so report that block rather than
    // the sink itself.
    if (live.empty())
    {
      for (const auto &[lid, g] : nodes)
        if (
          g.tag == "block" &&
          std::find(g.feeds.begin(), g.feeds.end(), sink_id) != g.feeds.end())
          throw UnsupportedConstructError(g.type_name, 2);
      throw UnsupportedConstructError("undriven sink " + target, 2);
    }

    for (int p : live)
      ensure_pf(p);

    if (live.size() == 1 && nodes.at(live.front()).tag != "leftPowerRail")
    {
      RungNode r = new_rung();
      r.elements.push_back(pf_contact(live.front()));
      r.elements.push_back(make_coil(target, kind));
      net.rungs.push_back(std::move(r));
      return;
    }

    // One accumulator per sink: two sinks driving the same variable (a
    // set/reset coil pair, say) must not share one.
    const std::string acc = synth_var(
      target + "__pf" + std::to_string(acc_counter++), VarKind::BOOL, true, 0);

    RungNode clear = new_rung();
    clear.elements.push_back(make_coil(acc, CoilKind::Reset));
    net.rungs.push_back(std::move(clear));

    for (int p : live)
    {
      RungNode r = new_rung();
      if (nodes.at(p).tag != "leftPowerRail")
        r.elements.push_back(pf_contact(p));
      r.elements.push_back(make_coil(acc, CoilKind::Set));
      net.rungs.push_back(std::move(r));
    }

    RungNode drive = new_rung();
    drive.elements.push_back(make_contact(acc, false, ContactEdge::None));
    drive.elements.push_back(make_coil(target, kind));
    net.rungs.push_back(std::move(drive));
  };

  // Emit a function block: first the rungs driving its enable pin, then the
  // FB step itself.  Blocks feeding this one are emitted first so that their
  // output pins are already assigned when this block reads them.
  std::set<int> emitted_blocks;
  emit_block = [&](int block_id) {
    if (!emitted_blocks.insert(block_id).second)
      return;

    const GNode &g = nodes.at(block_id);
    for (int p : live_preds(block_id))
      ensure_pf(p);

    FBKind kind;
    try
    {
      kind = fb_kind_of(g.type_name);
    }
    catch (const LdParseError &)
    {
      // User-defined FBs are executed from their Structured Text body
      // (UserFBInstance), not as a rung element.
      return;
    }

    const bool is_timer =
      kind == FBKind::TON || kind == FBKind::TOF || kind == FBKind::TP;
    const bool is_counter = kind == FBKind::CTU || kind == FBKind::CTD;
    if (!is_timer && !is_counter)
      return;

    const char *enable = is_timer ? "IN" : (kind == FBKind::CTU ? "CU" : "CD");
    const std::string enable_var =
      synth_var(pin_name(block_id, enable), VarKind::BOOL, true, 0);
    emit_sink(block_id, enable_var, CoilKind::Output);

    RungElement e;
    e.loc = loc;
    if (is_timer)
    {
      e.kind = RungElementKind::TimerFB;
      e.timer_fb.kind = kind;
      e.timer_fb.instance_name = inst_name(block_id);
      e.timer_fb.IN_var = enable_var;
      e.timer_fb.PT_var = resolve_data_pin(block_id, "PT", VarKind::INT);
      e.timer_fb.Q_var =
        synth_var(pin_name(block_id, "Q"), VarKind::BOOL, true, 0);
      e.timer_fb.ET_var =
        synth_var(pin_name(block_id, "ET"), VarKind::INT, true, 0);
      e.timer_fb.loc = loc;
    }
    else
    {
      e.kind = RungElementKind::CounterFB;
      e.counter_fb.kind = kind;
      e.counter_fb.instance_name = inst_name(block_id);
      if (kind == FBKind::CTU)
        e.counter_fb.CU_var = enable_var;
      else
        e.counter_fb.CD_var = enable_var;
      auto reset = g.in_pins.find("R");
      if (reset != g.in_pins.end() && nodes.count(reset->second))
      {
        if (!nodes.at(reset->second).var.empty())
          e.counter_fb.R_var = nodes.at(reset->second).var;
        else
          std::cerr << "warning: graphical LD: counter " << inst_name(block_id)
                    << " has its R pin driven by a contact chain, which is not "
                    << "modelled; the counter will not reset.\n";
      }
      e.counter_fb.PV_var = resolve_data_pin(block_id, "PV", VarKind::INT);
      e.counter_fb.Q_var =
        synth_var(pin_name(block_id, "Q"), VarKind::BOOL, true, 0);
      e.counter_fb.CV_var =
        synth_var(pin_name(block_id, "CV"), VarKind::INT, true, 0);
      e.counter_fb.loc = loc;
    }

    RungNode step = new_rung();
    step.elements.push_back(e);
    net.rungs.push_back(std::move(step));
  };

  // Step 6: snapshot the feedback variables before any rung runs.
  for (const auto &v : feedback_vars)
  {
    RungNode snap = new_rung();
    snap.elements.push_back(make_contact(v, false, ContactEdge::None));
    snap.elements.push_back(make_coil(
      synth_var(v + "__prev", VarKind::BOOL, true, 0), CoilKind::Output));
    net.rungs.push_back(std::move(snap));
  }

  // Step 7: emit one sink per coil, in rightPowerRail order — the order the
  // vendor tool draws the networks, hence the scan execution order.
  std::vector<int> coils;
  std::set<int> coils_seen;
  for (auto rpr : ld_body.children("rightPowerRail"))
    for (auto cpi : rpr.select_nodes(".//connection"))
    {
      int cid = cpi.node().attribute("refLocalId").as_int(-1);
      if (
        cid >= 0 && nodes.count(cid) && is_coil_tag(nodes.at(cid).tag) &&
        coils_seen.insert(cid).second)
        coils.push_back(cid);
    }
  for (auto &[lid, g] : nodes)
    if (is_coil_tag(g.tag) && coils_seen.insert(lid).second)
      coils.push_back(lid);

  for (int coil : coils)
  {
    const GNode &g = nodes.at(coil);
    CoilKind kind = CoilKind::Output;
    if (g.storage == "set")
      kind = CoilKind::Set;
    else if (g.storage == "reset")
      kind = CoilKind::Reset;
    emit_sink(coil, g.var, kind);
  }

  // Blocks whose outputs drive nothing still advance their internal state
  // every scan, so they are emitted even when no coil consumes them.
  for (auto &[lid, g] : nodes)
    if (g.tag == "block" || g.tag == "Block")
      emit_block(lid);

  return true;
}

NetworkNode PlcopenXmlParser::parse_network(const void *node_ptr)
{
  const auto &n = *static_cast<const pugi::xml_node *>(node_ptr);
  NetworkNode net;
  net.name = text_or_attr(n, "name", "name");
  net.loc = loc_from_node(n, source_file_);

  for (auto rung : n.children("rung"))
    net.rungs.push_back(parse_rung(&rung));
  for (auto rung : n.children("Rung"))
    net.rungs.push_back(parse_rung(&rung));

  // Graphical LD (tc6_0201): if no textual <rung> children were found,
  // attempt to extract rung logic from the connection graph.
  if (net.rungs.empty())
    parse_graphical_ld(n, net, source_file_, synth_vars_, scan_interval_ms_);

  return net;
}

// -----------------------------------------------------------------------
// Schema normalisation
// -----------------------------------------------------------------------

// Replace TIA Portal / Rockwell element names with canonical PLCopen names.
static void rename_vendor_tags(pugi::xml_node node)
{
  // Rockwell uses "contactNO" / "contactNC"; normalise to "contact" with negated attr.
  for (auto child : node.children())
  {
    std::string tag = child.name();
    if (tag == "contactNO")
      child.set_name("contact");
    else if (tag == "contactNC")
    {
      child.set_name("contact");
      child.append_attribute("negated").set_value("negated");
    }
    rename_vendor_tags(child);
  }
}

struct pugi_doc_wrapper
{
  pugi::xml_document doc;
};

void PlcopenXmlParser::normalise(pugi_doc_wrapper &w)
{
  rename_vendor_tags(w.doc.root());
}

// -----------------------------------------------------------------------
// Untranslated POU bodies
// -----------------------------------------------------------------------

// A body the front end does not translate leaves the scan cycle empty, so every
// property holds vacuously and the run reports a proof (#7354). Accept by
// whitelist, so a notation added to the schema later fails closed.
//
// Both the location and the notation have to match what parse() collects
// above: <addData> and <documentation> nest an unrelated <body> under the POU,
// and a ladder body under <transitions> is collected by nobody.
static void
reject_untranslated_bodies(const pugi::xml_node &root, const LdAst &ast)
{
  for (auto xpath_node :
       root.select_nodes("//pou/body/* | //pou/actions/action/body/* | "
                         "//pou/transitions/transition/body/*"))
  {
    const pugi::xml_node lang = xpath_node.node();
    const std::string tag = lang.name();

    const pugi::xml_node holder = lang.parent().parent();
    const std::string where = holder.name();
    const pugi::xml_node pou =
      where == "pou" ? holder : holder.parent().parent();
    const std::string pou_name = pou.attribute("name").as_string();

    if (
      (tag == "LD" || tag == "ladderDiagram") &&
      (where == "pou" || where == "action"))
      continue;

    // st_fb_translator inlines a function block's Structured Text body into
    // the scan cycle, but only once the definition has registered: an FB with
    // no output pin or an empty body is dropped, and dropping it silently is
    // the same defect as dropping the body outright.
    const bool translated_fb_body =
      tag == "ST" &&
      std::string(pou.attribute("pouType").as_string()) == "functionBlock" &&
      std::any_of(
        ast.user_fb_defs.begin(),
        ast.user_fb_defs.end(),
        [&pou_name](const UserFBDef &d) { return d.type_name == pou_name; });
    if (translated_fb_body)
      continue;

    const std::string site =
      where == "pou" ? "POU '" + pou_name + "'"
                     : where + " '" + holder.attribute("name").as_string() +
                         "' of POU '" + pou_name + "'";
    throw UnsupportedConstructError(tag + " body of " + site, 2);
  }
}

// -----------------------------------------------------------------------
// Top-level parse()
// -----------------------------------------------------------------------

LdAst PlcopenXmlParser::parse(const std::string &path)
{
  source_file_ = path;

  pugi_doc_wrapper w;
  pugi::xml_parse_result result = w.doc.load_file(path.c_str());
  if (!result)
    throw LdParseError(path + ": " + result.description());

  normalise(w);

  LdAst ast;
  ast.source_file = path;

  pugi::xml_node root = w.doc.document_element();

  // Detect interrupt tasks (Tier-2 rejection).
  // PLCopen XML interrupt tasks carry type="INTERRUPT" or taskType="INTERRUPT".
  // Periodic tasks have a priority attribute but type="CYCLIC" or no type.
  for (auto xpath :
       root.select_nodes("//task[@type='INTERRUPT' or @taskType='INTERRUPT']"))
  {
    (void)xpath;
    ast.has_interrupt_tasks = true;
    break;
  }

  if (ast.has_interrupt_tasks)
    throw UnsupportedConstructError("InterruptTask", 2);

  // The cyclic task period sets the tick length of the fixed-tick time model
  // (§3.3): one scan iteration advances time by exactly one interval.
  scan_interval_ms_ = 0;
  for (auto xpath : root.select_nodes("//task[@interval]"))
  {
    const long long ms =
      parse_duration_ms(xpath.node().attribute("interval").as_string(""));
    if (ms > 0)
    {
      scan_interval_ms_ = static_cast<unsigned>(ms);
      break;
    }
  }

  // Parse variable declarations (global + local)
  for (auto xpath_var : root.select_nodes(
         "//pou/interface//*[self::inputVars or self::outputVars or "
         "self::inOutVars or self::localVars or self::globalVars]"))
  {
    pugi::xml_node vars_node = xpath_var.node();
    std::string vars_tag = vars_node.name();
    for (auto var_node : vars_node.children("variable"))
    {
      VarDecl v = parse_var_decl(&var_node);
      if (vars_tag.find("input") != std::string::npos)
        v.is_input = true;
      if (vars_tag.find("output") != std::string::npos)
        v.is_output = true;
      ast.variables.push_back(std::move(v));
    }
  }

  // Parse networks (one per POU body, plus one per SFC/transition action
  // body). Beremiz and other vendor tools commonly place the rung-bearing
  // <LD> network inside <pou><actions><action><body><LD>, not directly
  // under <pou><body>, when the LD logic is invoked from an SFC step
  // action rather than forming the POU's own top-level body. Both
  // locations must be searched, or the action-nested rungs are silently
  // skipped and the program verifies vacuously (no rung assignments,
  // all variables at their zero-initialised default).
  for (auto xpath_node :
       root.select_nodes("//pou/body/LD | //pou/actions/action/body/LD"))
  {
    pugi::xml_node body_node = xpath_node.node();
    NetworkNode net = parse_network(&body_node);
    if (net.name.empty())
    {
      pugi::xml_node action_node = body_node.parent().parent();
      std::string action_name = (std::string)action_node.name() == "action"
                                  ? text_or_attr(action_node, "name", "name")
                                  : "";
      net.name = action_name.empty() ? "main" : action_name;
    }
    ast.networks.push_back(std::move(net));
  }
  for (auto xpath_node :
       root.select_nodes("//pou/body/ladderDiagram | "
                         "//pou/actions/action/body/ladderDiagram"))
  {
    pugi::xml_node body_node = xpath_node.node();
    NetworkNode net = parse_network(&body_node);
    if (net.name.empty())
    {
      pugi::xml_node action_node = body_node.parent().parent();
      std::string action_name = (std::string)action_node.name() == "action"
                                  ? text_or_attr(action_node, "name", "name")
                                  : "";
      net.name = action_name.empty() ? "main" : action_name;
    }
    ast.networks.push_back(std::move(net));
  }

  // Declare the pins and path accumulators the graphical resolver invented.
  // They are already marked as driven, so the inference below leaves them
  // alone rather than havocking them as physical inputs.
  for (auto &v : synth_vars_)
    ast.variables.push_back(std::move(v));
  synth_vars_.clear();

  // Heuristic I/O inference for graphical LD programs without hardware
  // addresses (%IX/%QX). Variables that appear only as contacts across all
  // networks are treated as inputs; variables that appear only as coils are
  // treated as outputs. This covers tutorial and simulation programs that
  // declare all variables as <localVars> without address attributes.
  {
    std::set<std::string> contact_vars, coil_vars;
    for (const auto &net : ast.networks)
      for (const auto &rung : net.rungs)
        for (const auto &elem : rung.elements)
        {
          if (elem.kind == RungElementKind::Contact)
            contact_vars.insert(elem.contact.variable);
          if (elem.kind == RungElementKind::Coil)
            coil_vars.insert(elem.coil.variable);
        }
    for (auto &v : ast.variables)
    {
      if (v.is_input || v.is_output)
        continue;
      if (contact_vars.count(v.name) && !coil_vars.count(v.name))
        v.is_input = true;
      else if (coil_vars.count(v.name) && !contact_vars.count(v.name))
        v.is_output = true;
    }
  }

  // -------------------------------------------------------------------
  // User-defined function-block bodies (e.g. EQ_0/GT_0/SUB_0 wrappers that
  // may carry Ladder Logic Bombs).  Register each functionBlock POU's
  // Structured Text body, and record every block instance referencing one so
  // the converter can execute the translated body in the scan cycle.  This
  // makes logic hidden inside FB bodies reachable to the verifier instead of
  // being dropped as an unsupported rung element.
  // -------------------------------------------------------------------
  std::function<void(const pugi::xml_node &, std::string &)> collect_text =
    [&](const pugi::xml_node &n, std::string &out) {
      for (pugi::xml_node c : n)
      {
        if (c.type() == pugi::node_pcdata || c.type() == pugi::node_cdata)
          out += c.value();
        else
          collect_text(c, out);
      }
    };

  for (auto xp : root.select_nodes("//pou[@pouType='functionBlock']"))
  {
    pugi::xml_node pou = xp.node();
    UserFBDef def;
    def.type_name = pou.attribute("name").as_string();
    if (def.type_name.empty())
      continue;
    // Resolve a <variable>'s declared type the same way parse_var_decl does:
    // <type> holds a single child whose tag is the type name (or "derived"
    // carrying a name attribute).  Mapping through var_kind_from_string (which
    // knows REAL) keeps REAL FB variables from collapsing to BOOL/INT.
    auto fb_var_kind = [this](const pugi::xml_node &v) -> VarKind {
      pugi::xml_node tnode = v.child("type");
      pugi::xml_node first = tnode.first_child();
      if (!first)
        return VarKind::INT; // numeric default for an untyped FB variable
      std::string tag = first.name();
      std::string type_str =
        (tag == "derived") ? first.attribute("name").as_string() : tag;
      return var_kind_from_string(type_str);
    };
    for (auto v : pou.select_nodes(".//interface/inputVars/variable"))
      def.input_vars.push_back(
        {v.node().attribute("name").as_string(), fb_var_kind(v.node())});
    for (auto v : pou.select_nodes(".//interface/localVars/variable"))
      def.local_vars.push_back(
        {v.node().attribute("name").as_string(), fb_var_kind(v.node())});
    if (auto ov = pou.select_node(".//interface/outputVars/variable").node())
    {
      def.output_var = ov.attribute("name").as_string();
      def.output_kind = fb_var_kind(ov);
    }
    pugi::xml_node st = pou.select_node(".//body/ST").node();
    if (!st)
      continue; // non-ST body (e.g. graphical FB) — not handled here
    collect_text(st, def.st_body);
    if (def.output_var.empty() || def.st_body.empty())
      continue;
    ast.user_fb_defs.push_back(std::move(def));
  }

  if (!ast.user_fb_defs.empty())
  {
    for (auto xp : root.select_nodes("//pou[@pouType='program']//block"))
    {
      pugi::xml_node blk = xp.node();
      std::string tn = blk.attribute("typeName").as_string();
      for (const auto &def : ast.user_fb_defs)
      {
        if (def.type_name != tn)
          continue;
        UserFBInstance inst;
        inst.type_name = tn;
        inst.instance_name = blk.attribute("instanceName").as_string();
        inst.block_id = blk.attribute("localId").as_string();
        inst.loc = {source_file_, 0, 0};
        ast.user_fb_instances.push_back(std::move(inst));
        break;
      }
    }

    // Wire FB output pins to the program variables that consume them: a program
    // <outVariable> with <connection refLocalId="<block>" formalParameter="<pin>">
    // means "<prog_var> := <fb_instance>.<pin>".  This propagates a (possibly
    // forged) FB output into the program so value/actuator-manipulation bombs
    // become observable, and makes the model faithful instead of vacuous.
    for (auto xp : root.select_nodes("//pou[@pouType='program']//outVariable"))
    {
      pugi::xml_node ov = xp.node();
      std::string pv = ov.child("expression").child_value();
      if (pv.empty())
        pv = ov.child("variable").child_value();
      pugi::xml_node conn = ov.select_node(".//connection").node();
      if (!conn || pv.empty())
        continue;
      std::string ref = conn.attribute("refLocalId").as_string();
      std::string pin = conn.attribute("formalParameter").as_string();
      if (pin.empty())
        continue;
      for (auto &inst : ast.user_fb_instances)
        if (inst.block_id == ref)
          inst.out_wires.push_back({pv, pin});
    }
  }

  // Last, so the function-block definitions above are already registered.
  reject_untranslated_bodies(root, ast);

  return ast;
}
