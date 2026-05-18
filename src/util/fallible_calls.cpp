#include <util/fallible_calls.h>

#include <cassert>
#include <irep2/irep2_utils.h>

const std::vector<fallible_call_t> &fallible_calls()
{
  static const std::vector<fallible_call_t> table = {
    // Memory allocation (non-lowered forms — `malloc` and `realloc` are
    // rewritten to sideeffect2t by the Clang frontend and never reach this
    // checker as FUNCTION_CALL instructions; `calloc` and `aligned_alloc`
    // remain as ordinary calls).
    {"calloc", success_kind::non_null},
    {"aligned_alloc", success_kind::non_null},
    // String duplication.
    {"strdup", success_kind::non_null},
    {"strndup", success_kind::non_null},
    // Stdio.
    {"fopen", success_kind::non_null},
    {"freopen", success_kind::non_null},
    {"tmpfile", success_kind::non_null},
    {"fdopen", success_kind::non_null},
    // Environment.
    {"getenv", success_kind::non_null},
    // POSIX I/O.
    {"read", success_kind::non_negative},
    {"write", success_kind::non_negative},
    {"recv", success_kind::non_negative},
    {"send", success_kind::non_negative},
    {"open", success_kind::non_negative},
    {"creat", success_kind::non_negative},
    // POSIX threads (success == 0).
    {"pthread_mutex_lock", success_kind::zero},
    {"pthread_mutex_unlock", success_kind::zero},
    {"pthread_mutex_trylock", success_kind::zero},
    {"pthread_create", success_kind::zero},
    {"pthread_join", success_kind::zero},
  };
  return table;
}

/// Strip ESBMC operational-model suffixes the pthread OM uses when
/// rewriting user-level calls (`pthread_mutex_lock` →
/// `pthread_mutex_lock_noassert` / `_nocheck`, etc.). Without this every
/// pthread entry on the whitelist would be dead — the goto-program
/// carries the suffixed name, not the C-level one. The suffix list must
/// stay in sync with `src/c2goto/library/pthread_lib.c`; the
/// `fallible_calls_test` Catch suite asserts every pthread_* whitelist
/// entry resolves through both suffixes, so drift trips that test.
static std::string_view strip_om_suffix(std::string_view name)
{
  static constexpr std::string_view suffixes[] = {"_noassert", "_nocheck"};
  for (std::string_view s : suffixes)
    if (
      name.size() > s.size() &&
      name.compare(name.size() - s.size(), s.size(), s) == 0)
      return name.substr(0, name.size() - s.size());
  return name;
}

const fallible_call_t *find_fallible(std::string_view name)
{
  const std::string_view canonical = strip_om_suffix(name);
  for (const fallible_call_t &fc : fallible_calls())
    if (fc.name == canonical)
      return &fc;
  return nullptr;
}

expr2tc success_predicate(success_kind kind, const expr2tc &ret_value)
{
  const expr2tc zero = gen_zero(ret_value->type);
  switch (kind)
  {
  case success_kind::non_null:
    return notequal2tc(ret_value, zero);
  case success_kind::non_negative:
    return greaterthanequal2tc(ret_value, zero);
  case success_kind::zero:
    return equality2tc(ret_value, zero);
  }
  assert(false && "unhandled success_kind");
  abort();
}
