#pragma once
#include <irep2/irep2_templates_types.h>

// Explicit instantiation for exprs.

#define expr_typedefs1(basename, superclass)                                   \
  template class esbmct::                                                      \
    expr_methods2<basename##2t, superclass, superclass::traits>;               \
  irep_typedefs1(basename, superclass)

#define expr_typedefs2(basename, superclass)                                   \
  template class esbmct::                                                      \
    expr_methods2<basename##2t, superclass, superclass::traits>;               \
  irep_typedefs2(basename, superclass)

#define expr_typedefs3(basename, superclass)                                   \
  template class esbmct::                                                      \
    expr_methods2<basename##2t, superclass, superclass::traits>;               \
  irep_typedefs3(basename, superclass)

#define expr_typedefs4(basename, superclass)                                   \
  template class esbmct::                                                      \
    expr_methods2<basename##2t, superclass, superclass::traits>;               \
  irep_typedefs4(basename, superclass)

#define expr_typedefs5(basename, superclass)                                   \
  template class esbmct::                                                      \
    expr_methods2<basename##2t, superclass, superclass::traits>;               \
  irep_typedefs5(basename, superclass)

#define expr_typedefs6(basename, superclass)                                   \
  template class esbmct::                                                      \
    expr_methods2<basename##2t, superclass, superclass::traits>;               \
  irep_typedefs6(basename, superclass)

#define expr_typedefs_empty(basename, superclass)                              \
  template class esbmct::                                                      \
    expr_methods2<basename##2t, superclass, superclass::traits>;               \
  template class esbmct::                                                      \
    irep_methods2<basename##2t, superclass, superclass::traits>;               \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;
