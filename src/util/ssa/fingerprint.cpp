#include <util/ssa/fingerprint.h>

#include <irep2/irep2_utils.h>

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <tuple>

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

/// ESBMC names a local `c:<file>@<character-offset>@F@<fn>@<var>`
/// (clang-c-frontend). The offset moves whenever anything textually earlier in
/// the file changes, so it has to go before a name can be compared across
/// edits. Returns the name with that segment removed; other shapes
/// (`c:@F@main`, `__ESBMC_alloc`) are returned unchanged.
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

/// Anonymous struct/union/enum types are named `..._at_<file>_<line>_<col>`
/// (clang-c-frontend), so a type's name moves with any edit above its
/// declaration exactly as a local's name does. Types are not reached by
/// Foreach_operand, so the suffix is interned here, on the serialised text.
void canonicalise_type_locations(
  std::string &text,
  std::map<std::string, unsigned> &ids)
{
  const std::string tag = "_at_";
  size_t pos = 0;
  while ((pos = text.find(tag, pos)) != std::string::npos)
  {
    size_t end = pos + tag.size();
    while (end < text.size() && !isspace(static_cast<unsigned char>(text[end])))
      ++end;

    const std::string loc = text.substr(pos, end - pos);
    const unsigned next = ids.size();
    const unsigned id = ids.emplace(loc, next).first->second;
    const std::string rep = tag + "#" + std::to_string(id);

    text.replace(pos, end - pos, rep);
    pos += rep.size();
  }
}

class normalisert
{
public:
  explicit normalisert(fingerprint_modet mode) : mode(mode)
  {
  }

  void operator()(expr2tc &e)
  {
    if (is_nil_expr(e))
      return;

    e->Foreach_operand([this](expr2tc &sub) { (*this)(sub); });

    if (!is_symbol2t(e))
      return;

    symbol2t &s = to_symbol2t(e);
    const sym_keyt k = key_of(s);

    if (mode == fingerprint_modet::full)
    {
      s.thename = irep_idt("v" + std::to_string(intern(ids, k)));
      s.rlevel = symbol_renaming_level::level1;
      s.level1_num = 0;
      s.level2_num = 0;
    }
    else
    {
      if (mode == fingerprint_modet::srcloc)
        s.thename = irep_idt(disambiguate(s.thename.as_string()));
      s.level1_num = 0;
      s.level2_num = intern(per_name[s.thename.as_string()], k);
    }
    s.thread_num = 0;
    s.node_num = 0;
  }

  /// Canonicalise location-bearing type names in \p text. No-op under `raw`
  /// and `counters`, which are the un-normalised baselines.
  void canonicalise_text(std::string &text)
  {
    if (mode == fingerprint_modet::srcloc || mode == fingerprint_modet::full)
      canonicalise_type_locations(text, type_locs);
  }

private:
  static unsigned intern(std::map<sym_keyt, unsigned> &m, const sym_keyt &k)
  {
    const unsigned next = m.size();
    return m.emplace(k, next).first->second;
  }

  /// Two locals in sibling scopes differ only in the stripped offset, so the
  /// stripped name is suffixed with a first-occurrence index to keep distinct
  /// symbols distinct.
  const std::string &disambiguate(const std::string &orig)
  {
    auto it = stripped.find(orig);
    if (it != stripped.end())
      return it->second;

    const std::string base = strip_source_position(orig);
    auto &n = stripped_count[base];
    return stripped.emplace(orig, base + "#" + std::to_string(n++))
      .first->second;
  }

  fingerprint_modet mode;
  std::map<sym_keyt, unsigned> ids;
  std::map<std::string, std::map<sym_keyt, unsigned>> per_name;
  std::map<std::string, std::string> stripped;
  std::map<std::string, unsigned> stripped_count;
  std::map<std::string, unsigned> type_locs;
};

void mix(uint64_t &h, uint64_t v)
{
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
}

/// FNV-1a over the bytes of \p s.
uint64_t hash_text(const std::string &s)
{
  uint64_t h = 0xcbf29ce484222325ULL;
  for (unsigned char c : s)
  {
    h ^= c;
    h *= 0x100000001b3ULL;
  }
  return h;
}

/// Digest one expression under \p n. The copy is what gets mutated: irep2's
/// copy-on-write detaches on the first write, so the equation being solved is
/// left untouched.
///
/// Digested from pretty()'s text, NOT from crc(): irep_idt::hash() returns the
/// string-pool index (irep_idt.h:172), which is interning-order dependent, so
/// crc() is only meaningful within one process. Any persistent key has to be
/// built from the characters of a name.
void digest_expr(uint64_t &h, const expr2tc &e, normalisert &n)
{
  if (is_nil_expr(e))
  {
    mix(h, 1);
    return;
  }

  expr2tc copy = e;
  n(copy);
  std::string text = copy->pretty(0);
  n.canonicalise_text(text);
  // Set ESBMC_FP_DEBUG to diff the normalised text of two runs; that is how
  // both position-bearing name forms above were found.
  if (getenv("ESBMC_FP_DEBUG"))
    std::cerr << "FP " << text << "\n";
  mix(h, hash_text(text));
}
} // namespace

uint64_t ssa_cone_digest(
  const symex_target_equationt::SSA_stepst &steps,
  fingerprint_modet mode)
{
  // One normaliser across the whole cone: canonical ids must agree between
  // steps, or two occurrences of the same symbol would digest differently.
  normalisert n(mode);
  uint64_t h = 0xcbf29ce484222325ULL;

  for (const auto &step : steps)
  {
    if (step.ignore)
      continue;

    mix(h, static_cast<uint64_t>(step.type));
    digest_expr(h, step.guard, n);
    digest_expr(h, step.cond, n);
  }

  return h;
}

size_t ssa_cone_size(const symex_target_equationt::SSA_stepst &steps)
{
  size_t n = 0;
  for (const auto &step : steps)
    if (!step.ignore)
      ++n;
  return n;
}
