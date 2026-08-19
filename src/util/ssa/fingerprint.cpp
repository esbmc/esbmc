#include <util/ssa/fingerprint.h>

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
const char *mode_name(fingerprint_modet m)
{
  switch (m)
  {
  case fingerprint_modet::raw:
    return "raw";
  case fingerprint_modet::counters:
    return "counters";
  case fingerprint_modet::srcloc:
    return "srcloc";
  case fingerprint_modet::full:
    return "full";
  }
  return "?";
}

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

class normalisert
{
public:
  explicit normalisert(fingerprint_modet mode) : mode(mode)
  {
  }

  /// Collect every symbol the cone mentions, then name them from that set
  /// alone. Numbering by order of first occurrence would make the result
  /// depend on the order symex happened to emit the steps in, which is not
  /// stable across unrelated edits.
  void collect(const expr2tc &e)
  {
    if (is_nil_expr(e))
      return;

    e->foreach_operand([this](const expr2tc &sub) { collect(sub); });

    if (is_symbol2t(e))
      seen.insert(key_of(to_symbol2t(e)));
  }

  /// Assign every collected symbol its canonical name. Ranking is by the
  /// original counters, so an edit that shifts them all by the same amount
  /// leaves the ranks -- and the digest -- unchanged.
  void assign_names()
  {
    std::map<std::string, std::vector<sym_keyt>> groups;
    for (const sym_keyt &k : seen)
      groups[group_key(k)].push_back(k);

    for (auto &[base, members] : groups)
    {
      std::sort(
        members.begin(),
        members.end(),
        [](const sym_keyt &a, const sym_keyt &b) {
          return std::tie(
                   std::get<1>(a),
                   std::get<2>(a),
                   std::get<3>(a),
                   std::get<4>(a),
                   std::get<5>(a),
                   std::get<0>(a)) <
                 std::tie(
                   std::get<1>(b),
                   std::get<2>(b),
                   std::get<3>(b),
                   std::get<4>(b),
                   std::get<5>(b),
                   std::get<0>(b));
        });

      // Two symbols that differ only in the stripped offset are ordered by it,
      // which a uniform shift preserves.
      std::stable_sort(
        members.begin(),
        members.end(),
        [](const sym_keyt &a, const sym_keyt &b) {
          return offset_of(std::get<0>(a)) < offset_of(std::get<0>(b));
        });

      for (size_t i = 0; i < members.size(); ++i)
        names[members[i]] = mode == fingerprint_modet::full
                              ? "v" + std::to_string(i)
                              : base + "#" + std::to_string(i);
    }
  }

  void operator()(expr2tc &e)
  {
    if (is_nil_expr(e) || mode == fingerprint_modet::raw)
      return;

    e->Foreach_operand([this](expr2tc &sub) { (*this)(sub); });

    if (!is_symbol2t(e))
      return;

    symbol2t &s = to_symbol2t(e);
    auto it = names.find(key_of(s));
    if (it == names.end())
      return;

    s.thename = irep_idt(it->second);
    if (mode == fingerprint_modet::full)
      s.rlevel = symbol_renaming_level::level1;
    s.level1_num = 0;
    s.level2_num = 0;
    s.thread_num = 0;
    s.node_num = 0;
  }

  /// Canonicalise location-bearing type names in \p text. No-op under `raw`
  /// and `counters`, which are the un-normalised baselines.
  fingerprint_modet get_mode() const
  {
    return mode;
  }

  void canonicalise_text(std::string &text)
  {
    if (mode == fingerprint_modet::srcloc || mode == fingerprint_modet::full)
      canonicalise_type_locations(text, type_locs);
  }

private:
  /// Symbols are numbered within a group, so the group key decides what a
  /// mode treats as "the same symbol under a different version".
  std::string group_key(const sym_keyt &k) const
  {
    switch (mode)
    {
    case fingerprint_modet::full:
      return std::string();
    case fingerprint_modet::srcloc:
      return strip_source_position(std::get<0>(k));
    default:
      return std::get<0>(k);
    }
  }

  fingerprint_modet mode;
  std::set<sym_keyt> seen;
  std::map<sym_keyt, std::string> names;
  std::map<std::string, unsigned> type_locs;
};

/// Digest one expression under \p n; the mutated copy leaves the equation
/// being solved untouched, since irep2 detaches on the first write.
///
/// From pretty() text, never crc(): irep_idt::hash() is the string-pool index
/// (irep_idt.h:172), so crc() only means anything within one process.
std::string normalised_text(const expr2tc &e, normalisert &n)
{
  if (is_nil_expr(e))
    return "nil";

  expr2tc copy = e;
  n(copy);
  std::string text = copy->pretty(0);
  n.canonicalise_text(text);
  // --verbosity fingerprint:debug diffs the text two runs digest, which is how
  // a mismatch between them gets diagnosed; each line names its mode.
  log_debug("fingerprint", "FP[{}] {}", mode_name(n.get_mode()), text);
  return text;
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

std::string ssa_cone_text(
  const symex_target_equationt::SSA_stepst &steps,
  fingerprint_modet mode)
{
  // One normaliser across the whole cone: canonical ids must agree between
  // steps, or two occurrences of the same symbol would render differently.
  normalisert n(mode);

  for (const auto &step : steps)
  {
    if (step.ignore)
      continue;
    n.collect(step.guard);
    n.collect(step.cond);
  }
  n.assign_names();

  std::vector<std::string> rendered;
  for (const auto &step : steps)
  {
    if (step.ignore)
      continue;

    rendered.push_back(
      "step " + std::to_string(static_cast<int>(step.type)) + "\n" +
      normalised_text(step.guard, n) + "\n" + normalised_text(step.cond, n) +
      "\n");
  }

  std::string out;
  for (const auto &step : rendered)
    out += step;

  return out;
}

uint64_t ssa_cone_digest(
  const symex_target_equationt::SSA_stepst &steps,
  fingerprint_modet mode)
{
  return fingerprint_hash(ssa_cone_text(steps, mode));
}

size_t ssa_cone_size(const symex_target_equationt::SSA_stepst &steps)
{
  size_t n = 0;
  for (const auto &step : steps)
    if (!step.ignore)
      ++n;
  return n;
}
