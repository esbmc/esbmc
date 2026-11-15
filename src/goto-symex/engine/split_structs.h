#pragma once

#include <goto-programs/goto_functions.h>
#include <irep2/irep2_expr.h>
#include <util/config/options.h>
#include <util/symtab/context.h>
#include <unordered_map>
#include <unordered_set>

/// The struct objects --field-sensitive-ssa gives one SSA name per field
/// (#7521). The set is decided once, before symex, and never changes: an
/// object is either always split or never split.
///
/// Field `k` of object `s` is the context symbol `s::k`; the index keeps
/// member names out of the id, since the value set reads `.` as a member
/// suffix and k-induction's havoc filters match on substrings of ids.
class split_structst
{
public:
  split_structst() = default;

  /// Pick the locals and parameters of fixed-size struct type that are only
  /// ever read whole or through a constant member, never by address or at
  /// byte level, and add their field symbols to @p context. Picks nothing
  /// unless --field-sensitive-ssa is set, or when the program may create
  /// threads or symex checks the stack size.
  void analyse(
    const goto_functionst &goto_functions,
    const namespacet &ns,
    contextt &context,
    const optionst &options,
    const std::unordered_set<irep_idt, irep_id_hash> &address_taken_locals);

  /// The L0 field symbols of @p name, or null when it is not split.
  const std::vector<expr2tc> *fields(const irep_idt &name) const
  {
    auto it = split.find(name);
    return it == split.end() ? nullptr : &it->second;
  }

  /// Whether @p name is a field symbol no value set entry can hold: a scalar
  /// narrower than a pointer.
  bool is_untracked_field(const irep_idt &name) const
  {
    return untracked_fields.count(name) != 0;
  }

  static bool is_untracked_scalar(const type2tc &type);

  bool empty() const
  {
    return split.empty();
  }

  /// @p e with every constant member of a split object replaced by its field
  /// symbol and every other occurrence of the object by the struct of its
  /// fields. Only L0 expressions taken from the program reach this.
  expr2tc rewrite(const expr2tc &e) const;

  /// The struct of @p sym's field symbols.
  expr2tc whole(const expr2tc &sym) const;

private:
  std::unordered_map<irep_idt, std::vector<expr2tc>, irep_id_hash> split;
  /// Member name to field index, per split object.
  std::unordered_map<
    irep_idt,
    std::unordered_map<irep_idt, unsigned, irep_id_hash>,
    irep_id_hash>
    index;
  std::unordered_set<irep_idt, irep_id_hash> untracked_fields;
};
