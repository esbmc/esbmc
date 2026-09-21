#pragma once

#include <esbmc/ts/transition_system.h>

#include <iosfwd>

class namespacet;
class optionst;

/// Write the transition system as a BTOR2 safety problem with a single bad
/// property, for external hardware model checkers. Throws std::runtime_error
/// on anything outside scalar bit-vector and boolean terms.
///
/// \p bitblast_fp opts into floating point, which BTOR2 has no sort for and
/// which therefore has to be lowered to bit-vector operations. The lowering is
/// generic softfloat, so the result is correct but unoptimised; the in-memory
/// engines (--ts-bmc, --ts-k-induction, --ts-pdr) handle floats without it.
void write_btor2(
  const transition_systemt &ts,
  std::ostream &out,
  bool bitblast_fp,
  const namespacet &ns,
  const optionst &options);
