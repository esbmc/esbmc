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
#include <vector>

namespace
{
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

/// Anonymous aggregates and variables are named
/// `... __anon_<what>_at_<file>_<function>_<line>_<column>`
/// (clang_c_convert.cpp:4893, 4935, 4966), and the position moves with any
/// edit above the declaration, so it is cut. The `__anon_` marker anchors the
/// cut: names reaching here are not re-disambiguated the way symbols are, so
/// an unanchored `_at_` search truncates ordinary user identifiers and merges
/// two cones that a claim can tell apart -- `s.val_at_a` and `s.val_at_b`
/// digested alike, and one file's proof discharged the other file's violated
/// claim (esbmc/esbmc#7143). Two anonymous types that still collide once their
/// positions are gone have the same members, so they encode alike anyway.
std::string strip_type_location(const std::string &name)
{
  const size_t anon = name.find("__anon_");
  if (anon == std::string::npos)
    return name;

  const size_t at = name.find("_at_", anon);
  return at == std::string::npos ? name : name.substr(0, at);
}

/// The `#<n>` L2 version symex appends to every SSA name. Cut like the
/// character offset: a cone's versions are re-indexed from zero, so a claim
/// does not digest differently merely because symex numbered more branches
/// before reaching it (esbmc/esbmc#7143).
std::string strip_ssa_version(const std::string &name)
{
  const size_t hash = name.rfind('#');
  // An unsigned counter is at most 10 digits; a longer run is not one, and
  // accumulating it would wrap the sort key.
  if (
    hash == std::string::npos || hash + 1 == name.size() ||
    name.size() - hash - 1 > 10)
    return name;

  for (size_t i = hash + 1; i < name.size(); ++i)
    if (!isdigit(static_cast<unsigned char>(name[i])))
      return name;

  return name.substr(0, hash);
}

/// The `<n>` out of a trailing `#<n>`, or 0.
unsigned ssa_version_of(const std::string &name)
{
  const std::string base = strip_ssa_version(name);
  if (base.size() == name.size())
    return 0;

  unsigned value = 0;
  for (size_t i = base.size() + 1; i < name.size(); ++i)
    value = value * 10 + (name[i] - '0');
  return value;
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
      seen.insert(to_symbol2t(e).get_symbol_name());
  }

  void assign_names()
  {
    if (mode == fingerprint_modet::raw)
      return;

    std::map<std::string, std::vector<std::string>> groups;
    for (const std::string &id : seen)
      groups[strip_ssa_version(strip_source_position(id))].push_back(id);

    for (auto &[base, members] : groups)
    {
      // Two locals that differ only in the stripped offset are ordered by it,
      // which an edit shifting every later offset equally preserves; the SSA
      // version orders the assignments to one name within the cone.
      std::sort(
        members.begin(),
        members.end(),
        [](const std::string &a, const std::string &b) {
          const auto ka = std::make_pair(offset_of(a), ssa_version_of(a));
          const auto kb = std::make_pair(offset_of(b), ssa_version_of(b));
          return ka != kb ? ka < kb : a < b;
        });

      for (size_t i = 0; i < members.size(); ++i)
        names[members[i]] = mode == fingerprint_modet::full
                              ? "v" + std::to_string(name_count++)
                              : base + '\x01' + std::to_string(i);
    }
  }

  /// The token a symbol contributes to the hash. The renaming level is
  /// appended because get_symbol_name() renders level0 and level1_global
  /// alike, and a cache key must not merge two distinct symbols.
  std::string rename_symbol(const symbol2t &sym) const
  {
    const std::string id = sym.get_symbol_name();
    auto it = names.find(id);
    return (it != names.end() ? it->second : id) + '/' +
           static_cast<char>('0' + static_cast<int>(sym.rlevel));
  }

  /// The form of a name reached outside a symbol node -- a type tag, a struct
  /// component, or the symbol named by symbol_type2t/code_decl2t. There is no
  /// per-cone identity to canonicalise these against, so each strip must be
  /// injective on its own: whatever it cuts has to be something no two names
  /// can differ by alone.
  std::string rename(const irep_idt &id) const
  {
    if (mode == fingerprint_modet::raw)
      return id.as_string();
    return strip_type_location(strip_source_position(id.as_string()));
  }

private:
  fingerprint_modet mode;
  std::set<std::string> seen;
  std::map<std::string, std::string> names;
  size_t name_count = 0;
};

/// The expressions convert_internal_step feeds the solver for \p step, in the
/// order it feeds them. `guard` and `cond` account for an assume, assert,
/// branching or assignment step -- for an assignment `cond` is the `lhs == rhs`
/// equality. A renumber carries the symbol and its new object size in
/// `lhs`/`rhs` and leaves `cond` nil, and an output's arguments live in its
/// payload; neither is reachable through `cond`, so neither may be left out of
/// the key (esbmc/esbmc#7143).
template <class visitort>
void for_each_encoded_expr(
  const symex_target_equationt::SSA_stept &step,
  const visitort &visit)
{
  visit(step.guard);
  visit(step.cond);

  if (step.is_renumber())
  {
    visit(step.lhs);
    visit(step.rhs);
  }
  else if (step.is_output() && step.output_data)
    for (const expr2tc &arg : step.output_data->output_args)
      visit(arg);
}

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
    for_each_encoded_expr(step, [&n](const expr2tc &e) { n.collect(e); });
  }
  n.assign_names();

  const irep2_name_mappert rename = [&n](const irep_idt &id) {
    return n.rename(id);
  };
  const irep2_symbol_mappert rename_symbol = [&n](const symbol2t &sym) {
    return n.rename_symbol(sym);
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
    for_each_encoded_expr(step, [&](const expr2tc &e) {
      irep2_content_hash(e, rename, rename_symbol, key.lo, key.hi);
    });
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
