#include <pointer-analysis/field_census.h>
#include <util/message/message.h>
#include <util/symtab/namespace.h>
#include <map>

namespace field_census
{
bool enabled = false;
const namespacet *ns = nullptr;

namespace
{
struct objectt
{
  bool is_union = false;
  bool address_taken = false;
  bool deref_field_path = false;
  bool byte_level = false;
  bool is_global = false;
  bool is_heap = false;
  bool nested = false;
  unsigned leaves = 0;
};

std::map<irep_idt, objectt> objects;
unsigned arrays_of_structs = 0;
std::map<irep_idt, bool> seen_arrays;

unsigned count_leaves(const type2tc &t, bool &nested)
{
  if (!is_struct_type(t))
    return 1;
  unsigned n = 0;
  for (const type2tc &m : to_struct_type(t).members)
  {
    if (is_struct_type(m) || is_union_type(m))
      nested = true;
    n += is_struct_type(m) ? count_leaves(m, nested) : 1;
  }
  return n;
}

const expr2tc *root_symbol(const expr2tc &e)
{
  const expr2tc *cur = &e;
  while (!is_nil_expr(*cur))
  {
    if (is_symbol2t(*cur))
      return cur;
    if (is_member2t(*cur))
      cur = &to_member2t(*cur).source_value;
    else if (is_index2t(*cur))
      cur = &to_index2t(*cur).source_value;
    else if (is_typecast2t(*cur))
      cur = &to_typecast2t(*cur).from;
    else
      return nullptr;
  }
  return nullptr;
}

objectt *lookup(const expr2tc &sym)
{
  if (!is_symbol2t(sym))
    return nullptr;
  const type2tc &t = sym->type;
  if (is_array_type(t) && is_struct_type(to_array_type(t).subtype))
  {
    if (!seen_arrays[to_symbol2t(sym).thename])
    {
      seen_arrays[to_symbol2t(sym).thename] = true;
      arrays_of_structs++;
    }
    return nullptr;
  }
  if (!is_struct_type(t) && !is_union_type(t))
    return nullptr;
  const symbol2t &s = to_symbol2t(sym);
  auto [it, fresh] = objects.try_emplace(s.thename);
  if (fresh)
  {
    objectt &o = it->second;
    o.is_union = is_union_type(t);
    const std::string name = s.thename.as_string();
    o.is_heap = name.rfind("symex_dynamic::", 0) == 0;
    const symbolt *base = ns ? ns->lookup(s.thename) : nullptr;
    o.is_global = base && base->static_lifetime;
    if (!o.is_union)
      o.leaves = count_leaves(t, o.nested);
  }
  return &it->second;
}

void walk(const expr2tc &e)
{
  if (is_nil_expr(e))
    return;
  if (is_symbol2t(e))
    lookup(e);
  else if (is_address_of2t(e))
  {
    if (const expr2tc *r = root_symbol(to_address_of2t(e).ptr_obj))
      if (objectt *o = lookup(*r))
        o->address_taken = true;
  }
  else if (is_byte_extract2t(e) || is_byte_update2t(e))
  {
    const expr2tc &src = is_byte_extract2t(e)
                           ? to_byte_extract2t(e).source_value
                           : to_byte_update2t(e).source_value;
    if (const expr2tc *r = root_symbol(src))
      if (objectt *o = lookup(*r))
        o->byte_level = true;
  }
  e->foreach_operand([](const expr2tc &op) { walk(op); });
}
} // namespace

void note_instruction(const goto_programt::instructiont &insn)
{
  walk(insn.code);
  walk(insn.guard);
}

bool reinterprets(const expr2tc &e)
{
  if (is_nil_expr(e))
    return false;
  if (is_byte_extract2t(e) || is_byte_update2t(e) || is_bitcast2t(e))
    return true;
  bool found = false;
  e->foreach_operand(
    [&found](const expr2tc &op) { found = found || reinterprets(op); });
  return found;
}

void note_deref(const expr2tc &root_object, bool field_path)
{
  const expr2tc *r = root_symbol(root_object);
  if (!r)
    return;
  objectt *o = lookup(*r);
  if (!o)
    return;
  o->address_taken = true;
  o->deref_field_path = true;
  if (!field_path)
    o->byte_level = true;
}

void report()
{
  unsigned never_taken = 0, field_paths = 0, byte_level = 0, unions = 0;
  unsigned globals = 0, heap = 0, nested = 0, max_leaves = 0;
  unsigned over_256 = 0, over_4096 = 0;
  for (const auto &[name, o] : objects)
  {
    if (o.is_union)
    {
      unions++;
      continue;
    }
    if (o.byte_level)
      byte_level++;
    else if (o.address_taken)
      field_paths++;
    else
      never_taken++;
    globals += o.is_global;
    heap += o.is_heap;
    nested += o.nested;
    max_leaves = std::max(max_leaves, o.leaves);
    over_256 += o.leaves > 256;
    over_4096 += o.leaves > 4096;
  }
  log_status(
    "field-census: objects={} never_address_taken={} const_field_paths={} "
    "byte_level={} union={} | struct globals={} heap={} nested={} "
    "max_leaves={} leaves>256={} leaves>4096={} arrays_of_structs={}",
    objects.size(),
    never_taken,
    field_paths,
    byte_level,
    unions,
    globals,
    heap,
    nested,
    max_leaves,
    over_256,
    over_4096,
    arrays_of_structs);
}
} // namespace field_census
