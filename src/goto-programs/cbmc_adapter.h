#pragma once

#include <goto-programs/read_cbmc_goto_object.h>
#include <util/irep.h>

#include <string>
#include <utility>
#include <vector>

/// Result of adapting a CBMC parse into ESBMC irep conventions. The ireps here
/// are ready to be consumed by symbolt::from_irep (symbols) and the
/// goto_program_irep convert() (functions), exactly as ESBMC's own goto-binary
/// reader feeds them.
struct cbmc_adapted_resultt
{
  std::vector<irept> symbols;
  std::vector<std::pair<std::string, irept>> functions;
};

/// Rewrites CBMC irep conventions into ESBMC's. Faithful port of
/// goto-transcoder's adapter.rs (ESBMCParseResult::from(CBMCParseResult)).
cbmc_adapted_resultt adapt_cbmc_to_esbmc(cbmc_parse_resultt parsed);

/// Maps a CBMC goto-program instruction-type value onto the matching ESBMC
/// goto_program_instruction_typet value.
///
/// CBMC and ESBMC share the `goto_program_instruction_typet` numbering for
/// every kind that reaches a finished straight-line/control-flow binary, so
/// those map by identity. The kinds that diverge are handled explicitly:
///   - START_THREAD (6) / END_THREAD (7): CBMC encodes concurrency as
///     instruction types, whereas ESBMC models it with `__ESBMC_spawn_thread`
///     intrinsic calls and has no instruction-type counterpart.
///   - INCOMPLETE_GOTO (19): an unresolved goto that should not survive into a
///     finished binary; ESBMC reuses 19 as a removed-enumerator gap.
/// These divergent kinds are rejected with a named diagnostic (the function
/// aborts) rather than being silently mis-cast in symex. \p cbmc_type is a raw
/// CBMC instruction-type value; the return value is the ESBMC enumerator.
unsigned map_cbmc_instruction_type(unsigned cbmc_type);
