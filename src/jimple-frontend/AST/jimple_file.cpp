#include <map>
#include <fstream>
#include <jimple-frontend/AST/jimple_file.h>
#include <util/std_code.h>
#include <util/expr_util.h>

// (De)-Serialization helpers (from jimple_ast.h)
void to_json(json &, const jimple_ast &)
{
  // Don't care
}

void from_json(const json &j, jimple_ast &p)
{
  p.from_json(j);
}

std::string jimple_file::to_string() const
{
  std::ostringstream oss;
  oss << "Jimple File:\n"
      << "\t"
      << "Name: " << this->class_name << "\n\t"
      << "Mode: " << to_string(this->mode) << "\n\t"
      << "Extends: " << this->extends << "\n\t"
      << "Implements: " << this->implements << "\n\t" << this->m.to_string();

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
  j.at("name").get_to(this->class_name);


  std::string t;
  j.at("object").get_to(t);
  this->mode = from_string(t);
    
  try
  {
    j.at("implements").get_to(this->implements);
  }
  catch(std::exception &e)
  {
    this->implements = "(No implements)";
  }

  try
  {
    j.at("extends").get_to(this->extends);
  }
  catch(std::exception &e)
  {
    this->implements = "(No extends)";
  }

  auto modifiers = j.at("modifiers");
  m = modifiers.get<jimple_modifiers>();
  
  auto filebody = j.at("content");  
  for(auto &x : filebody)
  {
    // TODO: Here is where to add support for signatures
    auto content_type = x.at("object").get<std::string>();
    if(content_type == "Method")
    {
      auto member = x.get<jimple_class_method>();
      body.push_back(member);
    }
  }
}

inline jimple_file::file_type
jimple_file::from_string(const std::string &name) const
{
  return from_map.at(name);
}

inline std::string
jimple_file::to_string(const jimple_file::file_type &ft) const
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
  /*
   * A class is just a type, this method will just register
   * its type and return a code_skipt.
   *
   * However, the list of symbols are going to be updated with
   * the static functions and variables, and constructor */

  exprt e = code_skipt();

  // TODO: support interface
  if(is_interface())
  {
    throw "Interface is not supported";
  }

  std::string id, name;
  id = "tag-" + this->getClassName();
  name = this->getClassName();

  // Check if class already exists
  if(ctx.find_symbol(id) != nullptr)
    throw "Duplicated class name";

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
    //t.components().push_back(comp);
  }

  added_symbol->type = t;
  return e;
}
