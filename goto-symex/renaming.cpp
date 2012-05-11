#include <irep2.h>
#include <migrate.h>

#include "renaming.h"

std::string renaming::level1t::name(const irep_idt &identifier,
                                    unsigned frame) const
{
  return id2string(identifier)+"@"+i2string(frame)+"!"+i2string(_thread_id);
}

unsigned renaming::level2t::current_number(
  const irep_idt &identifier) const
{
  current_namest::const_iterator it=current_names.find(identifier);
  if(it==current_names.end()) return 0;
  return it->second.count;
}

std::string renaming::level1t::get_ident_name(const irep_idt &identifier) const
{

  current_namest::const_iterator it=
    current_names.find(identifier);

  if(it==current_names.end())
  {
    // can not find
    return id2string(identifier); // means global value ?
  }

  return name(identifier, it->second);
}

std::string renaming::level2t::get_ident_name(const irep_idt &identifier) const
{
  current_namest::const_iterator it=
    current_names.find(identifier);

  if(it==current_names.end())
    return name(identifier, 0);

  return name(identifier, it->second.count);
}

std::string
renaming::level2t::name(const irep_idt &identifier, unsigned count) const
{
  unsigned int n_id = 0;
  current_namest::const_iterator it =current_names.find(identifier);
  if(it != current_names.end())
    n_id = it->second.node_id;
  return id2string(identifier)+"&"+i2string(n_id)+"#"+i2string(count);
}

void renaming::level1t::rename(exprt &expr)
{
  // rename all the symbols with their last known value

  if(expr.id()==exprt::symbol)
  {
    const irep_idt &identifier=expr.identifier();

    // first see if it's already an l1 name

    if(identifier.as_string().find("@") != std::string::npos)
      return;

    const current_namest::const_iterator it=
      current_names.find(identifier);

    if(it!=current_names.end())
      expr.identifier(name(identifier, it->second));
  }
  else if(expr.id()==exprt::addrof ||
          expr.id()=="implicit_address_of" ||
          expr.id()=="reference_to")
  {
    assert(expr.operands().size()==1);
    rename(expr.op0());
  }
  else
  {
    // do this recursively
    Forall_operands(it, expr)
      rename(*it);
  }
}

void renaming::level2t::rename(exprt &expr)
{

  expr2tc theexpr;
  migrate_expr(expr, theexpr);
  renaming::level2t::rename(theexpr);
  expr = migrate_expr_back(theexpr);
  return;
}

void renaming::level2t::rename(expr2tc &expr)
{
  // rename all the symbols with their last known value

  if (is_symbol2t(expr))
  {
    symbol2t &sym = to_symbol2t(expr);

    // first see if it's already an l2 name

    if (sym.name.as_string().find("#") != std::string::npos)
      return;

    const current_namest::const_iterator it = current_names.find(sym.name);

    if(it!=current_names.end())
    {
      if (!is_nil_expr(it->second.constant))
        expr = it->second.constant; // sym is now invalid reference
      else
        sym.name = name(sym.name, it->second.count);
    }
    else
    {
      std::string new_identifier = name(sym.name, 0);
      sym.name = irep_idt(new_identifier);
    }
  }
  else if (is_address_of2t(expr))
  {
    // do nothing
  }
  else
  {
    // do this recursively
    std::vector<expr2tc*> operands;
    expr.get()->list_operands(operands);
    for (std::vector<expr2tc*>::iterator it = operands.begin();
         it != operands.end(); it++)
      rename(**it);
  }
}

void renaming::level2t::coveredinbees(const irep_idt &identifier, unsigned count, unsigned node_id)
{
  valuet &entry=current_names[identifier];
  entry.count=count;
  entry.node_id = node_id;
}

void renaming::renaming_levelt::get_original_name(exprt &expr) const
{
  Forall_operands(it, expr)
    get_original_name(*it);

  if(expr.id()==exprt::symbol)
  {
    irep_idt ident = get_original_name(expr.identifier());
    expr.identifier(ident);
  }
}

const irep_idt renaming::renaming_levelt::get_original_name(
  const irep_idt &identifier, std::string idxchar) const
{
  std::string namestr = identifier.as_string();

  // If this is renamed at all, it'll have the suffix:
  //   @x!y&z#n
  // So to undo this, find and remove everything after @, if it exists.
  size_t pos = namestr.find(idxchar);
  if (pos == std::string::npos)
    return identifier; // It's not named at all.

  // Remove suffix
  namestr = namestr.substr(0, pos);
  return irep_idt(namestr);
}

void renaming::level1t::print(std::ostream &out) const
{
  for(current_namest::const_iterator
      it=current_names.begin();
      it!=current_names.end();
      it++)
    out << it->first << " --> "
        << name(it->first, it->second) << std::endl;
}

void renaming::level2t::print(std::ostream &out) const
{
  for(current_namest::const_iterator
      it=current_names.begin();
      it!=current_names.end();
      it++)
    out << it->first << " --> "
        << name(it->first, it->second.count) << std::endl;

}

void renaming::level2t::dump() const
{

  print(std::cout);
}

irep_idt
renaming::level2t::make_assignment(irep_idt l1_ident, const exprt &const_value,
                           const exprt &assigned_value __attribute__((unused)))
{
  irep_idt new_name;

  valuet &entry = current_names[l1_ident];

  // This'll update entry beneath our feet; could reengineer it in the future.
  rename(l1_ident, entry.count + 1);

  new_name = name(l1_ident, entry.count);

  migrate_expr(const_value, entry.constant);

  return new_name;
}
