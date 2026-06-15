#include <ld-frontend/parser/plcopen_xml_parser.h>
#include <pugixml.hpp>
#include <cassert>
#include <unordered_map>
#include <functional>
#include <set>

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

static LdLocation
loc_from_node(const pugi::xml_node &n, const std::string &file)
{
  LdLocation loc;
  loc.file = file;
  // PLCopen XML does not standardise source coordinates; use addData if present.
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
    {"TIME", VarKind::TIME},
  };
  auto it = table.find(s);
  if (it == table.end())
    return VarKind::BOOL; // default; type checker will flag unsupported types
  return it->second;
}

ContactKind PlcopenXmlParser::contact_kind_from_string(const std::string &s)
{
  if (s == "negated" || s == "NormallyClosedContact")
    return ContactKind::NormallyClosed;
  return ContactKind::NormallyOpen;
}

CoilKind PlcopenXmlParser::coil_kind_from_string(const std::string &s)
{
  if (s == "set" || s == "SetCoil")
    return CoilKind::Set;
  if (s == "reset" || s == "ResetCoil")
    return CoilKind::Reset;
  return CoilKind::Output;
}

FBKind PlcopenXmlParser::fb_kind_from_string(const std::string &s)
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
    elem.contact.kind = contact_kind_from_string(
      text_or_attr(n, "negated", nullptr).empty() ? tag : "negated");
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
// <connection refLocalId="..."/> children.  The textual <rung> wrapper
// is absent.  We resolve the graph with a DFS from every leftPowerRail
// to every coil, convert each rail-to-coil path into a series contact
// chain (AND), and combine parallel paths to the same coil with OR by
// emitting multiple single-element rungs (the GOTO-IR emitter already
// OR-combines multiple rungs that write the same output coil).
//
// Returns true if the LD body contained graphical elements and rungs
// were successfully extracted; false if this is a textual LD body.

struct GNode
{
  std::string tag;        // "contact", "coil", "leftPowerRail", "block", ...
  std::string var;        // variable name (contacts and coils)
  bool negated = false;   // normally-closed contact
  std::string storage;    // "set", "reset", or "" for normal coil
  std::vector<int> feeds; // forward edges (this node feeds these localIds)
};

static bool parse_graphical_ld(
  const pugi::xml_node &ld_body,
  NetworkNode &net,
  const std::string &source_file)
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
    // Variable name
    if (auto v = child.child("variable"))
      g.var = v.child_value();
    // Negated contact
    std::string neg_attr = child.attribute("negated").as_string("");
    g.negated = (neg_attr == "true" || neg_attr == "negated");
    // Coil storage kind
    std::string storage_attr = child.attribute("storage").as_string("");
    if (storage_attr.empty())
    {
      if (t == "SetCoil")
        storage_attr = "set";
      if (t == "ResetCoil")
        storage_attr = "reset";
    }
    g.storage = storage_attr;
    // Back-edges: <connection refLocalId="X"/> means X feeds this node
    // We collect them now and build forward edges below.
    nodes[lid] = g;
  }

  // Step 3: build forward edges from backward (refLocalId) edges.
  // <connection refLocalId="X"/> is nested inside <connectionPointIn>,
  // so we use select_nodes to find all <connection> descendants.
  for (auto child : ld_body.children())
  {
    int lid = child.attribute("localId").as_int(-1);
    if (lid < 0 || nodes.find(lid) == nodes.end())
      continue;
    for (auto conn_node : child.select_nodes(".//connection"))
    {
      auto conn = conn_node.node();
      int src = conn.attribute("refLocalId").as_int(-1);
      if (src >= 0 && nodes.count(src))
        nodes[src].feeds.push_back(lid);
    }
  }

  // Step 4: DFS from every leftPowerRail to every coil.
  // Each path becomes one RungNode with one contact per step + one coil.
  int rung_counter = 0;

  std::function<void(int, int, std::vector<int> &, std::set<int> &)> dfs =
    [&](int cur, int target, std::vector<int> &path, std::set<int> &visited) {
      if (cur == target)
      {
        // Emit one RungNode for this path.
        // path contains nodes from leftRail up to (not including) target.
        // We append target here so the coil is included.
        RungNode rung;
        rung.id = "g" + std::to_string(rung_counter++);
        rung.loc = {source_file, 0, 0};

        // Emit contacts from the path
        for (int nid : path)
        {
          const GNode &g = nodes.at(nid);
          if (g.tag != "contact")
            continue;
          RungElement elem;
          elem.loc = {source_file, 0, 0};
          elem.kind = RungElementKind::Contact;
          elem.contact.kind =
            g.negated ? ContactKind::NormallyClosed : ContactKind::NormallyOpen;
          elem.contact.variable = g.var;
          elem.contact.loc = elem.loc;
          rung.elements.push_back(elem);
        }

        // Emit the coil (target node)
        {
          const GNode &g = nodes.at(target);
          RungElement elem;
          elem.loc = {source_file, 0, 0};
          elem.kind = RungElementKind::Coil;
          if (g.storage == "set")
            elem.coil.kind = CoilKind::Set;
          else if (g.storage == "reset")
            elem.coil.kind = CoilKind::Reset;
          else
            elem.coil.kind = CoilKind::Output;
          elem.coil.variable = g.var;
          elem.coil.loc = elem.loc;
          rung.elements.push_back(elem);
        }

        if (!rung.elements.empty())
          net.rungs.push_back(rung);
        return;
      }

      if (visited.count(cur))
        return;
      visited.insert(cur);
      path.push_back(cur);

      for (int nxt : nodes.at(cur).feeds)
        dfs(nxt, target, path, visited);

      path.pop_back();
      visited.erase(cur);
    };

  // Find all left rails
  std::vector<int> left_rails;
  for (auto &[lid, g] : nodes)
    if (g.tag == "leftPowerRail")
      left_rails.push_back(lid);

  // Collect coils in rightPowerRail order (defines scan execution order).
  // The rightPowerRail's <connectionPointIn> children list the coils in order.
  std::vector<int> coils;
  std::set<int> coils_seen;
  for (auto rpr : ld_body.children("rightPowerRail"))
  {
    for (auto cpi : rpr.select_nodes(".//connection"))
    {
      int cid = cpi.node().attribute("refLocalId").as_int(-1);
      if (
        cid >= 0 && nodes.count(cid) &&
        (nodes.at(cid).tag == "coil" || nodes.at(cid).tag == "SetCoil" ||
         nodes.at(cid).tag == "ResetCoil") &&
        !coils_seen.count(cid))
      {
        coils.push_back(cid);
        coils_seen.insert(cid);
      }
    }
  }
  // Add any coils not connected to rightPowerRail
  for (auto &[lid, g] : nodes)
    if (
      (g.tag == "coil" || g.tag == "SetCoil" || g.tag == "ResetCoil") &&
      !coils_seen.count(lid))
      coils.push_back(lid);

  for (int rail : left_rails)
    for (int coil : coils)
    {
      std::vector<int> path;
      std::set<int> visited;
      // Start from each node that the rail feeds
      for (int nxt : nodes.at(rail).feeds)
        dfs(nxt, coil, path, visited);
    }

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
    parse_graphical_ld(n, net, source_file_);

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

  // Parse networks (one per POU body)
  for (auto xpath_node : root.select_nodes("//pou/body/LD"))
  {
    pugi::xml_node body_node = xpath_node.node();
    NetworkNode net = parse_network(&body_node);
    if (net.name.empty())
      net.name = "main";
    ast.networks.push_back(std::move(net));
  }
  for (auto xpath_node : root.select_nodes("//pou/body/ladderDiagram"))
  {
    pugi::xml_node body_node = xpath_node.node();
    NetworkNode net = parse_network(&body_node);
    if (net.name.empty())
      net.name = "main";
    ast.networks.push_back(std::move(net));
  }

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

  return ast;
}
