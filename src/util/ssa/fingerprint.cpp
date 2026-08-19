#include <util/ssa/fingerprint.h>

#include <fmt/format.h>
#include <irep2/irep2_content_hash.h>
#include <irep2/irep2_utils.h>
#include <util/message/message.h>

#include <cctype>
#include <algorithm>
#include <map>
#include <string>
#include <set>
#include <tuple>
#include <vector>

namespace
{
/// (base name, level, l1, l2, thread, node) — a symbol's full SSA identity.
using sym_keyt =
  std::tuple<std::string, int, unsigned, unsigned, unsigned, unsigned>;

sym_keyt key_of(const symbol2t &s)
{
  return {
    s.thename.as_string(),
    static_cast<int>(s.rlevel),
    s.level1_num,
    s.level2_num,
    s.thread_num,
    s.node_num};
}

/// Drops the `<file>@<character-offset>@` of a local's name
/// `c:<file>@<offset>@F@<fn>@<var>`: the offset moves whenever anything
/// textually earlier changes. Other shapes are returned unchanged.
std::string strip_source_position(const std::string &name)
{
  const size_t colon = name.find(':');
  if (colon == std::string::npos)
    return name;

  const size_t at1 = name.find('@', colon);
  if (at1 == std::string::npos)
    return name;

  const size_t at2 = name.find('@', at1 + 1);
  if (at2 == std::string::npos || at2 == at1 + 1)
    return name;

  for (size_t i = at1 + 1; i < at2; ++i)
    if (!isdigit(static_cast<unsigned char>(name[i])))
      return name;

  return name.substr(0, colon + 1) + name.substr(at2 + 1);
}

/// Anonymous types are named `..._at_<file>_<line>_<col>`, which moves with
/// any edit above the declaration. Types are not reached by Foreach_operand,
/// so the suffix is interned here, on the serialised text.

/// The character offset out of `c:<file>@<offset>@...`, or 0.
unsigned offset_of(const std::string &name)
{
  const size_t colon = name.find(':');
  if (colon == std::string::npos)
    return 0;
  const size_t at1 = name.find('@', colon);
  if (at1 == std::string::npos)
    return 0;
  unsigned value = 0;
  for (size_t i = at1 + 1; i < name.size() && isdigit((unsigned char)name[i]);
       ++i)
    value = value * 10 + (name[i] - '0');
  return value;
}

/// Anonymous types are named `..._at_<file>_<line>_<col>`, which moves with any
/// edit above the declaration. The position is cut; the type's fields still
/// distinguish it.
std::string strip_type_location(const std::string &name)
{
  const size_t at = name.find("_at_");
  return at == std::string::npos ? name : name.substr(0, at);
}

class normalisert
{
public:
  explicit normalisert(fingerprint_modet mode) : mode(mode)
  {
  }

  /// Collect every symbol name the cone mentions. Naming from that set rather
  /// than by order of first occurrence keeps the result independent of the
  /// order symex happened to emit the steps in.
  void collect(const expr2tc &e)
  {
    if (is_nil_expr(e))
      return;

    e->foreach_operand([this](const expr2tc &sub) { collect(sub); });

    if (is_symbol2t(e))
      seen.insert(to_symbol2t(e).thename);
  }

  void assign_names()
  {
    if (mode == fingerprint_modet::raw)
      return;

    std::map<std::string, std::vector<irep_idt>> groups;
    for (const irep_idt &id : seen)
      groups[strip_source_position(id.as_string())].push_back(id);

    for (auto &[base, members] : groups)
    {
      // Two locals that differ only in the stripped offset are ordered by it,
      // which an edit shifting every later offset equally preserves.
      std::sort(
        members.begin(),
        members.end(),
        [](const irep_idt &a, const irep_idt &b) {
          return offset_of(a.as_string()) < offset_of(b.as_string());
        });

      for (size_t i = 0; i < members.size(); ++i)
        names[members[i]] = mode == fingerprint_modet::full
                              ? "v" + std::to_string(name_count++)
                              : base + "#" + std::to_string(i);
    }
  }

  /// The name a symbol contributes to the hash.
  std::string rename(const irep_idt &id) const
  {
    auto it = names.find(id);
    if (it != names.end())
      return it->second;
    return mode == fingerprint_modet::raw ? id.as_string()
                                          : strip_type_location(id.as_string());
  }

private:
  fingerprint_modet mode;
  std::set<irep_idt> seen;
  std::map<irep_idt, std::string> names;
  size_t name_count = 0;
};

} // namespace

uint64_t fingerprint_hash(const std::string &s)
{
  uint64_t h = 0xcbf29ce484222325ULL;
  for (unsigned char c : s)
  {
    h ^= c;
    h *= 0x100000001b3ULL;
  }
  return h;
}

ssa_cone_keyt ssa_cone_key(
  const symex_target_equationt::SSA_stepst &steps,
  fingerprint_modet mode)
{
  normalisert n(mode);
  for (const auto &step : steps)
  {
    if (step.ignore)
      continue;
    n.collect(step.guard);
    n.collect(step.cond);
  }
  n.assign_names();

  const irep2_name_mappert rename = [&n](const irep_idt &id) {
    return n.rename(id);
  };

  // Steps are fed in equation order: convert_internal_step encodes a claim as
  // implies(assumpt_expr, cond) over the assumes seen before it, so the
  // sequence is part of what the claim means.
  ssa_cone_keyt key{0xcbf29ce484222325ULL, 0x9e3779b97f4a7c15ULL};
  for (const auto &step : steps)
  {
    if (step.ignore)
      continue;

    key.lo ^= static_cast<uint64_t>(step.type);
    key.lo *= 0x100000001b3ULL;
    irep2_content_hash(step.guard, rename, key.lo, key.hi);
    irep2_content_hash(step.cond, rename, key.lo, key.hi);
  }
  return key;
}

std::string ssa_cone_key_string(
  const symex_target_equationt::SSA_stepst &steps,
  fingerprint_modet mode)
{
  const ssa_cone_keyt k = ssa_cone_key(steps, mode);
  return fmt::format("{:016x}{:016x}", k.lo, k.hi);
}

uint64_t ssa_cone_digest(
  const symex_target_equationt::SSA_stepst &steps,
  fingerprint_modet mode)
{
  return ssa_cone_key(steps, mode).lo;
}

size_t ssa_cone_size(const symex_target_equationt::SSA_stepst &steps)
{
  size_t n = 0;
  for (const auto &step : steps)
    if (!step.ignore)
      ++n;
  return n;
}
