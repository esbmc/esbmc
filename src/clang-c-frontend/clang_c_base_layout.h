#ifndef ESBMC_CLANG_C_FRONTEND_CLANG_C_BASE_LAYOUT_H
#define ESBMC_CLANG_C_FRONTEND_CLANG_C_BASE_LAYOUT_H

#include <util/arith/mp_arith.h>
#include <util/irep/std_types.h>
#include <util/symtab/namespace.h>

/// Displacement of \p base_id's subobject inside \p derived, in bytes, under
/// whichever layout ESBMC gave the class. False when neither layout places the
/// base at a single fixed displacement -- a virtual base shared by two sibling
/// bases has none.
///
/// The one oracle for this question. Shared rather than file-local because the
/// IREP2 adjust pass needs the same answer, and taking the offset from anywhere
/// else is the mistake scope-clang-c-irep2.md \#3894 records
/// (docs/roadmap/scope-clang-cpp-irep2.md \S3.12).
bool base_displacement(
  const namespacet &ns,
  const typet &derived,
  const irep_idt &base_id,
  BigInt &offset);

#endif
