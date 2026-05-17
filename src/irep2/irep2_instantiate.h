#pragma once

// Generates the explicit out-of-line instantiation of irep_methods2 (and
// its expr_methods2 / type_methods2 derivatives) for a single irep node.
// One instantiation per node; this header is a thin macro layer that
// names the concrete template arguments.

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

// Empty-field variants: identical to the regular macros — the field
// walk just iterates zero times. Retained as named aliases so call
// sites can document intent ("this node has no user fields").
#define ESBMC_INSTANTIATE_TYPE_EMPTY(basename, superclass)                     \
  ESBMC_INSTANTIATE_TYPE(basename, superclass)

#define ESBMC_INSTANTIATE_EXPR_EMPTY(basename, superclass)                     \
  ESBMC_INSTANTIATE_EXPR(basename, superclass)
