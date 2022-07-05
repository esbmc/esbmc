#include <map>
#include <fstream>
#include <jimple-frontend/AST/jimple_file.h>
#include <util/std_code.h>
#include <util/expr_util.h>

// Deserialization helpers
void to_json(json&, const jimple_ast&) {
  // Don't care
}

void from_json(const json& j, jimple_ast& p) {
  p.from_json(j);
}

std::string jimple_file::to_string() const
{
  std::ostringstream oss;
  oss << "Jimple File:\n"
      << "\t" << "Name: " << this->class_name
      << "\n\t" << "Mode: " << to_string(this->mode)
      << "\n\t" << "Extends: " << this->extends
      << "\n\t" << "Implements: " << this->implements
      << "\n\t" << this->m.to_string();

  oss << "\n\n";
  for(auto &x : body)
  {
    oss << x.to_string();
    oss << "\n\n";
  }

  return oss.str();
}

void jimple_file::from_json(const json &j)
{
  // Get ClassName
  j.at("classname").get_to(this->class_name);

  std::string t;
  j.at("filetype").get_to(t);
  this->mode = from_string(t);

  try {
    j.at("implements").get_to(this->implements);
  }
  catch(std::exception &e)
  {
    this->implements = "(No implements)";
  }

  try {
    j.at("extends").get_to(this->extends);
  }
  catch(std::exception &e)
  {
    this->implements = "(No extends)";
  }

  auto modifiers = j.at("modifiers");
  m = modifiers.get<jimple_modifiers>();

  auto filebody = j.at("filebody");
  for(auto &x : filebody)
  {
    // TODO: Here is where to add support for signatures
    if(x.contains("method"))
    {
      auto member = x.at("method").get<jimple_class_method>();
      body.push_back(member);
    }
  }
}

namespace {
typedef jimple_file::file_type file_type;
std::map<std::string, file_type> from_map = {
  {"Class", file_type::Class},
  {"Interface", file_type::Interface}};

std::map<file_type, std::string> to_map = {
  {file_type::Class, "Class"},
  {file_type::Interface, "Interface"}};
}
jimple_file::file_type jimple_file::from_string(const std::string &name)
{
  return from_map.at(name);
}
std::string jimple_file::to_string(const jimple_file::file_type &ft) const
{
  return to_map.at(ft);
}
void jimple_file::load_file(const std::string &path)
{
  std::ifstream i(path);
  json j;
  i >> j;

  from_json(j);
}

exprt jimple_file::to_exprt(contextt &ctx) const
{
  exprt e = code_skipt();

  if(is_interface())
  {
    throw "Interface is not supported";
  }

  std::string id, name;
  id = this->getClassName();
  name = this->getClassName();

  if(ctx.find_symbol(id) != nullptr)
    throw "Duplicated symbol";

  struct_typet t;
  t.tag(name);


  auto symbol = create_jimple_symbolt(t, name, name, id);

  std::string symbol_name = symbol.id.as_string();

  // A class/interface is a type
  symbol.is_type = true;

  // Add symbol into the context
  ctx.move_symbol_to_context(symbol);
  symbolt *added_symbol = ctx.find_symbol(symbol_name);

  // Add class/interface members
  for(auto const &field : body)
  {
    struct_typet::componentt comp;
    exprt &tmp = comp;
    tmp = field.to_exprt(ctx, name, name);
    // TODO: only add declarations
    t.components().push_back(comp);
  }

  added_symbol->type = t;
  return e;
}
