#include <irep2.h>
#include <migrate.h>
#include <prefix.h>

#include "renaming.h"

std::string renaming::level1t::name(const irep_idt &identifier,
                                    unsigned frame) const
{
  return id2string(identifier)+"@"+i2string(frame)+"!"+i2string(_thread_id);
}

unsigned renaming::level2t::current_number(const expr2tc &symbol) const
{
  current_namest::const_iterator it=current_names.find(symbol);
  if(it==current_names.end()) return 0;
  return it->second.count;
}

void
renaming::level1t::get_ident_name(expr2tc &sym) const
{
  symbol2t &symbol = to_symbol2t(sym);

  current_namest::const_iterator it =
    current_names.find(symbol.get_symbol_name());

  if (it == current_names.end())
  {
    // can not find; it's a global symbol.
    symbol.rlevel = symbol2t::level1_global;
    return;
  }

  symbol.rlevel = symbol2t::level1;
  symbol.level1_num = it->second;
  symbol.thread_num = _thread_id;
  return;
}

void
renaming::level2t::get_ident_name(expr2tc &sym) const
{
  symbol2t &symbol = to_symbol2t(sym);

  current_namest::const_iterator it = current_names.find(sym);

  symbol2t::renaming_level lev = symbol.rlevel =
              (symbol.rlevel == symbol2t::level1) ? symbol2t::level2
                                                  : symbol2t::level2_global;

  if (it == current_names.end()) {
    // Un-numbered so far.
    symbol.rlevel = lev;
    symbol.level2_num = 0;
    symbol.node_num = 0;
    return;
  }

  symbol.rlevel = lev;
  symbol.level2_num = it->second.count;
  symbol.node_num = it->second.node_id;
  return;
}

std::string
renaming::level2t::name(const irep_idt &identifier __attribute__((unused)), unsigned count __attribute__((unused))) const
{
  std::cerr << "renaming::level2t::name now dying" << std::endl;
  abort();
}

void renaming::level1t::rename(expr2tc &expr) const
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

void renaming::level2t::rename(expr2tc &expr) const
{
  // rename all the symbols with their last known value

  if (is_symbol2t(expr))
  {
    symbol2t &sym = to_symbol2t(expr);

    // first see if it's already an l2 name

    if (sym.rlevel == symbol2t::level2 || sym.rlevel == symbol2t::level2_global)
      return;

    if (sym.thename == "NULL")
      return;
    if (sym.thename == "INVALID")
      return;
    if (has_prefix(sym.thename.as_string(), "nondet$"))
      return;

    const current_namest::const_iterator it = current_names.find(expr);

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

void renaming::level2t::coveredinbees(expr2tc &lhs_sym, unsigned count, unsigned node_id)
{
#ifndef NDEBUG
  symbol2t &sym = to_symbol2t(lhs_sym);
  assert(sym.rlevel == symbol2t::level1 ||
         sym.rlevel == symbol2t::level1_global);
#endif

  valuet &entry=current_names[lhs_sym];
  entry.count=count;
  entry.node_id = node_id;
}

void renaming::renaming_levelt::get_original_name(expr2tc &expr,
                                            symbol2t::renaming_level lev) const
{

  Forall_operands2(it, oper_list, expr)
    get_original_name(**it);

  if (is_symbol2t(expr))
  {
    symbol2t &sym = to_symbol2t(expr);

    // Rename level2_global down to level1_global, not level1
    if (lev == symbol2t::level1 && sym.rlevel == symbol2t::level2_global)
      lev = symbol2t::level1_global;

    // Can't rename any lower,
    if (sym.rlevel == symbol2t::level0)
      return;

    // Wipe out some data with default values and set renaming level to whatever
    // was requested.
    switch (lev) {
    case symbol2t::level1:
    case symbol2t::level1_global:
      sym.rlevel = lev;
      sym.node_num = 0;
      sym.level2_num = 0;
      return;
    case symbol2t::level0:
      sym.rlevel = lev;
      sym.node_num = 0;
      sym.level2_num = 0;
      sym.thread_num = 0;
      sym.level1_num = 0;
      return;
    default:
      std::cerr << "get_original_nameing to invalid level " << lev << std::endl;
      abort();
    }
  }
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
      it++) {
    assert(to_symbol2t(it->first).rlevel == symbol2t::level1 || to_symbol2t(it->first).rlevel == symbol2t::level1_global);
    out << to_symbol2t(it->first).get_symbol_name() << " --> ";
    expr2tc tmp = it->first;
    rename(tmp);
    out << to_symbol2t(tmp).get_symbol_name() << std::endl;
  } }

void renaming::level2t::dump() const
{

  print(std::cout);
}

void
renaming::level2t::make_assignment(expr2tc &lhs_symbol,
                                   const expr2tc &const_value,
                          const expr2tc &assigned_value __attribute__((unused)))
{
  irep_idt new_name;

  assert(to_symbol2t(lhs_symbol).rlevel == symbol2t::level1 ||
         to_symbol2t(lhs_symbol).rlevel == symbol2t::level1_global);
  valuet &entry = current_names[lhs_symbol];

  // This'll update entry beneath our feet; could reengineer it in the future.
  rename(lhs_symbol, entry.count + 1);

  symbol2t &symbol = to_symbol2t(lhs_symbol);
  symbol2t::renaming_level lev = (symbol.rlevel == symbol2t::level0 ||
                                  symbol.rlevel == symbol2t::level1_global)
                                  ? symbol2t::level2_global : symbol2t::level2;
  symbol.rlevel = lev;
  // These fields were updated by the rename call,
  symbol.level2_num = entry.count; 
  symbol.node_num = entry.node_id;

  entry.constant = const_value;
}
