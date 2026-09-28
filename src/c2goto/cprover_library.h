#pragma once

#include <string>
#include <vector>
#include <util/irep/irep.h>

class languaget;
class contextt;
class goto_functionst;

/* Registers the bundled internal libc headers and sources with
 * file_operations. */
void register_bundled_libc();

/* Returns the path the headers of the internal libc have been extracted to
 * or NULL if no library is configured (either via config.ansi_c.lib or during
 * build time). */
const std::string *internal_libc_header_dir();

/* Adds the internal libc to `context` by parsing and linking all C sources.
 *
 * Note that parsing the entire ESBMC standard library is a slow process.
 */
void add_bundled_library_sources(
  contextt &context,
  const languaget &c_language);

void add_cprover_library(
  contextt &context,
  const languaget *language = nullptr);

/// Record that \p id came from a bundled operational-model library rather than
/// from the program. Loaders outside this file (the Python model blob) call it
/// so prune_unreferenced_library_functions can tell the two apart by provenance
/// instead of by guessing from the symbol name.
void record_library_symbol(const irep_idt &id);

/// Erase every library FUNCTION symbol not reachable from the program's own
/// symbols, so goto_convert never lowers it.
///
/// Frontends whose converter resolves model calls by name (Python) cannot use
/// the declaration-driven filter in add_cprover_library, because at load time
/// the context holds no program yet: they link the models wholesale and call
/// this once the converter has run. Only code symbols are erased -- types stay,
/// so the closed-world exception registry (exception_typeidt) is unchanged.
///
/// Pair it with assert_no_pruned_calls after goto_convert: a pass that runs
/// between the two can introduce a reference this closure could not see, and
/// that must fail loudly rather than leave a bodyless function returning
/// nondet.
///
/// \p prelowered carries the bodies of a library shipped as GOTO (the Python
/// model blob), whose symbols reach the context with nil values: without it the
/// walk sees no edge out of a model function and erases everything it calls.
void prune_unreferenced_library_functions(
  contextt &context,
  const goto_functionst *prelowered = nullptr);

/// Abort if any call in \p functions targets a symbol the prune erased.
void assert_no_pruned_calls(const goto_functionst &functions);
