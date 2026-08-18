#pragma once

#include <util/symtab/context.h>
#include <util/irep/irep.h>

#include <string>

/// Callee-name matching shared by `clang_c_adjust::do_special_functions` and
/// its IREP2 counterpart. Both lower the same builtins, so both must agree on
/// which spellings are one; a second copy of these predicates is exactly the
/// shape of divergence `unit/util/c_typecast.test.cpp` exists to catch
/// elsewhere.

/// `name`, `namef`, `named`, `namel`.
bool compare_float_suffix(const irep_idt &identifier, const std::string &name);

/// `name` and its `__`- and `__builtin_`-prefixed spellings, each with the
/// float suffixes: `isnan`, `__isnanf`, `__builtin_isnanl`, ...
bool compare_unscore_builtin(
  const irep_idt &identifier,
  const std::string &name);

/// True for the abs builtins that may be lowered to an `abs` node. That node
/// becomes `(x >= 0) ? x : -x`, ill-typed for anything but an arithmetic
/// argument, so a program overloading the name for a class type --
/// std::abs(complex) is why <complex> ships without it -- keeps its call.
bool is_abs_builtin_name(const irep_idt &identifier);

/// The lowerings that match a callee's *base* name, so a program that defines
/// one of these names itself would have its body discarded and the builtin
/// verified in its place (#6904). These are all spellings a program is free to
/// reuse -- `mylib::abs`, `mylib::isinf` -- unlike the `__builtin_`-prefixed
/// and CPROVER-prefixed entries, which are reserved.
bool is_name_matched_builtin(const irep_idt &identifier);

/// True when lowering this call would throw away a definition the program
/// supplies. Libc's own declarations are bodiless and the <cmath> overloads
/// forward to their `__builtin_` spelling, so both still lower.
/// @param base_name the spelling the lowerings match on
/// @param symbol_id the callee's linkage identifier, which is what the symbol
///        table is keyed by
bool builtin_shadows_user_definition(
  const contextt &context,
  const irep_idt &base_name,
  const irep_idt &symbol_id);
