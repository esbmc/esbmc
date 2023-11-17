/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_scopes.h>
#include <util/i2string.h>

cpp_scopet &cpp_scopest::new_block_scope()
{
  unsigned prefix = ++current_scope().compound_counter;
  cpp_scopet &n = new_scope(i2string(prefix));
  n.id_class = cpp_idt::BLOCK_SCOPE;
  return n;
}

void cpp_scopest::get_ids(
  const irep_idt &base_name,
  id_sett &id_set,
  bool current_only)
{
  id_set.clear();

  if (current_only)
  {
    current_scope().lookup(base_name, id_set);
    return;
  }

  current_scope().recursive_lookup(base_name, id_set);
}

void cpp_scopest::get_ids(
  const irep_idt &base_name,
  cpp_idt::id_classt id_class,
  id_sett &id_set,
  bool current_only)
{
  id_set.clear();

  if (current_only)
  {
    current_scope().lookup(base_name, id_class, id_set);
    return;
  }

  current_scope().recursive_lookup(base_name, id_class, id_set);
}

cpp_idt &cpp_scopest::put_into_scope(
  const symbolt &symbol,
  cpp_scopet &scope,
  bool is_friend)
{
  assert(!symbol.id.empty());
  assert(!symbol.name.empty());

  // functions are also scopes
  if (symbol.type.id() == "code")
  {
    cpp_scopest::id_mapt::iterator id_it = id_map.find(symbol.id);
    if (id_it == id_map.end())
    {
      irep_idt block_base_name(std::string("$block:") + symbol.name.c_str());
      cpp_idt &id = scope.insert(block_base_name);
      id.id_class = cpp_idt::BLOCK_SCOPE;
      id.identifier = symbol.id;
      id.is_scope = true;
      id.prefix = id2string(scope.prefix) + id2string(symbol.name) + "::";
      id_map[symbol.id] = &id;
    }
  }

  if (is_friend)
  {
    cpp_save_scopet saved_scope(*this);
    go_to(scope);
    go_to_global_scope();

    cpp_idt &id = current_scope().insert(symbol.name);
    id.identifier = symbol.id;
    id.id_class = cpp_idt::SYMBOL;
    if (id_map.find(symbol.id) == id_map.end())
      id_map[symbol.id] = &id;
    return id;
  }

  cpp_idt &id = scope.insert(symbol.name);
  id.identifier = symbol.id;
  id.id_class = cpp_idt::SYMBOL;
  if (id_map.find(symbol.id) == id_map.end())
    id_map[symbol.id] = &id;
  return id;
}

void cpp_scopest::print_current(std::ostream &out) const
{
  const cpp_scopet *scope = current_scope_ptr;

  do
  {
    scope->print_fields(out);
    out << std::endl;
    scope = &scope->get_parent();
  } while (!scope->is_root_scope());
}
