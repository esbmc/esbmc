#pragma once

// Generates the explicit out-of-line instantiation of irep_methods2 (and
// its expr_methods2 / type_methods2 derivatives) for a single irep node.
//
// Background. The historic design wrapped each node in a recursive chain
// of irep_methods2 inheritance, one level per field, and forced the
// compiler to emit method bodies for every level via a hand-unrolled
// irep_typedefsN(basename, super) macro family. F1 collapsed that chain
// into a single non-recursive irep_methods2 class whose methods walk the
// trait list via fold expressions. There is exactly one instantiation
// per node now, so this header is a thin macro layer that names the
// concrete template arguments. No Boost.PP loops, no chain levels, no
// per-call nfields parameter.

#include <irep2/irep2_meta_templates.h>

#define ESBMC_INSTANTIATE_IREP_METHODS(basename, superclass)                   \
  template class esbmct::                                                      \
    irep_methods2<basename##2t, superclass, typename superclass::traits>

#define ESBMC_INSTANTIATE_TYPE(basename, superclass)                           \
  template class esbmct::                                                      \
    type_methods2<basename##2t, superclass, typename superclass::traits>;      \
  ESBMC_INSTANTIATE_IREP_METHODS(basename, superclass)

#define ESBMC_INSTANTIATE_EXPR(basename, superclass)                           \
  template class esbmct::                                                      \
    expr_methods2<basename##2t, superclass, typename superclass::traits>;      \
  ESBMC_INSTANTIATE_IREP_METHODS(basename, superclass)

// Empty-field variants kept for parity with the historic surface even
// though, with the flat design, they are identical to the regular
// macros — the field walk just iterates zero times. Existing call sites
// are migrated to the regular macros in the same change.
#define ESBMC_INSTANTIATE_TYPE_EMPTY(basename, superclass)                     \
  ESBMC_INSTANTIATE_TYPE(basename, superclass)

#define ESBMC_INSTANTIATE_EXPR_EMPTY(basename, superclass)                     \
  ESBMC_INSTANTIATE_EXPR(basename, superclass)
