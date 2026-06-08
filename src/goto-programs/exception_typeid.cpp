#include <goto-programs/exception_typeid.h>

#include <util/namespace.h>
#include <util/context.h>
#include <util/symbol.h>

#include <algorithm>

namespace
{
const std::string tag_prefix = "tag-";

/// "tag-Foo" -> "Foo"; anything without the prefix is returned unchanged.
irep_idt strip_tag(const irep_idt &id)
{
  const std::string &s = id.as_string();
  if (s.compare(0, tag_prefix.size(), tag_prefix) == 0)
    return irep_idt(s.substr(tag_prefix.size()));
  return id;
}
} // namespace

exception_typeidt::exception_typeidt(const namespacet &ns)
{
  // Seed each type symbol's direct bases from the "bases" metadata. This is
  // needed where a THROW list is not the full ancestry — e.g. the Python
  // frontend stops at Exception, so catch(BaseException) needs the
  // Exception->BaseException link from the symbol table. (C++ throw lists are
  // already fully flattened, so for them register_chain alone suffices.)
  std::set<irep_idt> names;
  ns.get_context().foreach_operand_in_order([&](const symbolt &s) {
    if (!s.is_type)
      return;
    const irep_idt name = strip_tag(s.id);
    names.insert(name);
    std::vector<irep_idt> bases;
    const irept &base_list = s.get_type().find("bases");
    for (const irept &base : base_list.get_sub())
      bases.push_back(strip_tag(base.id()));
    direct_bases.emplace(name, std::move(bases));
  });
  for (const irep_idt &name : names)
    name_to_id.emplace(name, next_id++);
  registered_count = name_to_id.size();
}

void exception_typeidt::register_chain(const std::vector<irep_idt> &chain)
{
  if (chain.empty())
    return;

  // Give every name an id.
  for (const irep_idt &name : chain)
    if (name_to_id.emplace(name, next_id).second)
    {
      ++next_id;
      ++registered_count;
    }

  // Only the dynamic type (front) is a subtype of the rest. The frontend emits
  // a *flattened* ancestry: `struct D : A, B` throws as [D, A, B], where A and
  // B are independent bases of D — NOT A <: B. Recording the tail as a linear
  // chain would invent false subtype relations (a later `throw A()` could then
  // wrongly match `catch (B&)`). Each dynamic type registers its own ancestry
  // from its own throw, so the relation stays complete without that inference.
  std::vector<irep_idt> &bases = direct_bases[chain.front()];
  for (std::size_t j = 1; j < chain.size(); ++j)
    if (std::find(bases.begin(), bases.end(), chain[j]) == bases.end())
      bases.push_back(chain[j]);
}

unsigned exception_typeidt::id_of(const irep_idt &name)
{
  auto it = name_to_id.find(name);
  if (it != name_to_id.end())
    return it->second;
  // Unknown (opaque/library) type: assign a fresh id so it still dispatches.
  return name_to_id.emplace(name, next_id++).first->second;
}

bool exception_typeidt::is_subtype(
  const irep_idt &thrown,
  const irep_idt &caught) const
{
  // catch (void*) catches any thrown pointer (names end in "_ptr").
  if (caught == "void_ptr")
  {
    const std::string &s = thrown.as_string();
    return s == "void_ptr" ||
           (s.size() >= 4 && s.compare(s.size() - 4, 4, "_ptr") == 0);
  }

  // Reflexive-transitive walk over the bases graph, guarding against cycles.
  std::set<irep_idt> seen;
  std::vector<irep_idt> worklist{thrown};

  while (!worklist.empty())
  {
    const irep_idt cur = worklist.back();
    worklist.pop_back();

    if (cur == caught)
      return true;
    if (!seen.insert(cur).second)
      continue;

    auto it = direct_bases.find(cur);
    if (it != direct_bases.end())
      worklist.insert(worklist.end(), it->second.begin(), it->second.end());
  }
  return false;
}

std::set<unsigned>
exception_typeidt::concrete_subtype_ids(const irep_idt &caught) const
{
  std::set<unsigned> ids;
  for (const auto &[name, id] : name_to_id)
    if (is_subtype(name, caught))
      ids.insert(id);
  return ids;
}

irep_idt exception_typeidt::name_of(unsigned id) const
{
  for (const auto &[name, candidate] : name_to_id)
  {
    if (candidate == id)
      return name;
  }

  return irep_idt();
}
