#include <ld-frontend/parser/plcopen_xml_parser.h>
#include <pugixml.hpp>
#include <cassert>
#include <unordered_map>

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

static LdLocation loc_from_node(const pugi::xml_node &n, const std::string &file)
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
    {"TON", FBKind::TON}, {"TOF", FBKind::TOF}, {"TP", FBKind::TP},
    {"CTU", FBKind::CTU}, {"CTD", FBKind::CTD},
    {"ADD", FBKind::ADD}, {"SUB", FBKind::SUB},
    {"MUL", FBKind::MUL}, {"DIV", FBKind::DIV},
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
  // <type><derived name="..."/> or <type><BOOL/> etc.
  auto type_node = n.child("type");
  std::string type_str;
  if (!type_node.first_child().empty())
    type_str = type_node.first_child().name();
  if (type_str.empty())
    type_str = "BOOL";
  v.kind = var_kind_from_string(type_str);
  v.loc = loc_from_node(n, source_file_);
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

  if (tag == "contact" || tag == "Contact" || tag == "NormallyOpenContact" ||
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
    elem.coil.kind = coil_kind_from_string(tag);
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

  // Unknown element — skip silently; type checker will validate completeness
  elem.kind = RungElementKind::Contact; // placeholder
  return elem;
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
    rung.elements.push_back(parse_rung_element(&child));
  return rung;
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

struct pugi_doc_wrapper { pugi::xml_document doc; };

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
  for (auto xpath : root.select_nodes("//task[@type='INTERRUPT' or @taskType='INTERRUPT']"))
  {
    (void)xpath;
    ast.has_interrupt_tasks = true;
    break;
  }

  if (ast.has_interrupt_tasks)
    throw UnsupportedConstructError("InterruptTask", 2);

  // Parse variable declarations (global + local)
  for (auto xpath_var :
       root.select_nodes("//pou/interface//*[self::inputVars or self::outputVars or "
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

  return ast;
}
