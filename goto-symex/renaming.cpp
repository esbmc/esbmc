#include <irep2.h>
#include <migrate.h>
#include <prefix.h>

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

void renaming::level1t::rename(expr2tc &expr)
{
  // rename all the symbols with their last known value

  if (is_nil_expr(expr))
    return;

  if (is_symbol2t(expr))
  {
    symbol2t &sym = to_symbol2t(expr);

    // first see if it's already an l1 name

    if (sym.rlevel != symbol2t::level0)
      return;

    const current_namest::const_iterator it =
      current_names.find(sym.get_symbol_name());

    if(it!=current_names.end())
      expr = expr2tc(new symbol2t(sym.type, sym.thename, symbol2t::level1,
                                  it->second, 0, _thread_id, 0));
  }
  else if (is_address_of2t(expr))
  {
    rename(to_address_of2t(expr).ptr_obj);
  }
  else
  {
    // do this recursively
    Forall_operands2(it, oper_list, expr)
      rename(**it);
  }
}

void renaming::level2t::rename(expr2tc &expr)
{
  // rename all the symbols with their last known value

  if (is_symbol2t(expr))
  {
    symbol2t &sym = to_symbol2t(expr);

    // first see if it's already an l2 name

    if (sym.rlevel == symbol2t::level2)
      return;

    if (sym.thename == "NULL")
      return;
    if (sym.thename == "INVALID")
      return;
    if (has_prefix(sym.thename.as_string(), "nondet$"))
      return;

    const current_namest::const_iterator it =
      current_names.find(sym.get_symbol_name());

    if(it!=current_names.end())
    {
      // Is this a global symbol? Gets renamed differently.
      symbol2t::renaming_level lev;
      if (sym.rlevel == symbol2t::level0)
        lev = symbol2t::level2_global;
      else
        lev = symbol2t::level2;

      if (!is_nil_expr(it->second.constant))
        expr = it->second.constant; // sym is now invalid reference
      else
        expr = expr2tc(new symbol2t(sym.type, sym.thename, lev,
                                    sym.level1_num, it->second.count,
                                    sym.thread_num, it->second.node_id));
    }
    else
    {
      symbol2t::renaming_level lev;
      if (sym.rlevel == symbol2t::level0 ||
          sym.rlevel == symbol2t::level1_global)
        lev = symbol2t::level2_global;
      else
        lev = symbol2t::level2;

      expr = expr2tc(new symbol2t(sym.type, sym.thename, lev,
                                  sym.level1_num, 0, sym.thread_num, 0));
    }
  }
  else if (is_address_of2t(expr))
  {
    // do nothing
  }
  else
  {
    // do this recursively
    Forall_operands2(it, oper_list, expr)
      rename(**it);
  }
}

void renaming::level2t::coveredinbees(const irep_idt &identifier, unsigned count, unsigned node_id)
{
  valuet &entry=current_names[identifier];
  entry.count=count;
  entry.node_id = node_id;
}

void renaming::renaming_levelt::get_original_name(expr2tc &expr) const
{

  Forall_operands2(it, oper_list, expr)
    get_original_name(**it);

  if (is_symbol2t(expr))
  {
    symbol2t &sym = to_symbol2t(expr);
    irep_idt ident = get_original_name(sym.get_symbol_name());
    expr = expr2tc(new symbol2t(sym.type, irep_idt(ident)));
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
renaming::level2t::make_assignment(irep_idt l1_ident,
                                   const expr2tc &const_value,
                          const expr2tc &assigned_value __attribute__((unused)))
{
  irep_idt new_name;

  valuet &entry = current_names[l1_ident];

  // This'll update entry beneath our feet; could reengineer it in the future.
  rename(l1_ident, entry.count + 1);

  new_name = name(l1_ident, entry.count);

  entry.constant = const_value;

  return new_name;
}
