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
