#pragma once

// Generates explicit out-of-line instantiations of the irep_methods2
// inheritance chain for a single irep node.
//
// Why this exists. The recursive chain `irep_methods2<derived, base, traits,
// fields>` derives from `irep_methods2<..., mp_pop_front<fields>>` and so on
// down to an empty-fields specialisation. The compiler will happily build the
// vtables but routinely *omits* the out-of-line method bodies for the
// intermediate levels — the cause of the historical link failures that the
// hand-written `irep_typedefs0..6` macros (now retired) were chasing.
//
// To force emission we explicitly instantiate every level of the chain. The
// number of levels equals `num_user_fields + 1`: the full traits, then one
// for each successive `mp_drop_c<fields, I>` until only the base portion of
// the field list remains (the empty-fields specialisation handles that
// terminating case implicitly).
//
// Both the historic and current call sites already know `num_user_fields`,
// so we keep it explicit and verify it against `traits::num_fields` with a
// static_assert (catching B9: field-count drift between the macro
// invocation and the per-node trait list).
//
// Usage at file scope:
//   ESBMC_INSTANTIATE_IREP_CHAIN(struct_type, struct_union_data, 5)
//   ESBMC_INSTANTIATE_TYPE(struct_type, struct_union_data, 5)
//   ESBMC_INSTANTIATE_EXPR(constant_struct, constant_datatype_data, 1)
//
// For nodes with no user fields beyond the base layout, use the *_EMPTY
// variants below.

#include <boost/mp11/algorithm.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/inc.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

#include <irep2/irep2_meta_templates.h>

// One instantiation: irep_methods2<derived, super, super::traits,
// mp_drop_c<super::traits::fields, i>>. Driven by BOOST_PP_REPEAT, which
// passes (z, i, data) where data is the (basename, superclass) tuple.
#define _ESBMC_IREP_INSTANTIATE_LEVEL(z, i, basename_super)                    \
  template class esbmct::irep_methods2<                                        \
    BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(2, 0, basename_super), 2t),               \
    BOOST_PP_TUPLE_ELEM(2, 1, basename_super),                                 \
    BOOST_PP_TUPLE_ELEM(2, 1, basename_super)::traits,                         \
    boost::mp11::mp_drop_c<                                                    \
      typename BOOST_PP_TUPLE_ELEM(2, 1, basename_super)::traits::fields,      \
      i>>;

// Both variants take `nfields` = the total length of
// superclass::traits::fields, i.e. the same value as
// superclass::traits::num_fields. That count is the number of
// irep_methods2 chain levels we need to instantiate to cover the full
// recursion from the complete field list down to the singleton list.
//
// Spelling the count at the call site keeps it visible in source review;
// the static_assert guards against drift between the spelled value and
// the actual trait list length (B9).

// Type variant: emits the outermost type_methods2 instantiation followed
// by `nfields` levels of the irep_methods2 chain.
#define ESBMC_INSTANTIATE_TYPE(basename, superclass, nfields)                  \
  static_assert(                                                               \
    boost::mp11::mp_size<superclass::traits::fields>::value ==                 \
      static_cast<std::size_t>(nfields),                                       \
    "ESBMC_INSTANTIATE_TYPE(" #basename                                        \
    "): nfields disagrees with superclass::traits::num_fields");               \
  template class esbmct::type_methods2<                                        \
    BOOST_PP_CAT(basename, 2t),                                                \
    superclass,                                                                \
    typename superclass::traits>;                                              \
  BOOST_PP_REPEAT(                                                             \
    nfields,                                                                   \
    _ESBMC_IREP_INSTANTIATE_LEVEL,                                             \
    (basename, superclass))

// Expr variant: emits the outermost expr_methods2 instantiation followed
// by `nfields` levels of the irep_methods2 chain.
#define ESBMC_INSTANTIATE_EXPR(basename, superclass, nfields)                  \
  static_assert(                                                               \
    boost::mp11::mp_size<superclass::traits::fields>::value ==                 \
      static_cast<std::size_t>(nfields),                                       \
    "ESBMC_INSTANTIATE_EXPR(" #basename                                        \
    "): nfields disagrees with superclass::traits::num_fields");               \
  template class esbmct::expr_methods2<                                        \
    BOOST_PP_CAT(basename, 2t),                                                \
    superclass,                                                                \
    superclass::traits>;                                                       \
  BOOST_PP_REPEAT(                                                             \
    nfields,                                                                   \
    _ESBMC_IREP_INSTANTIATE_LEVEL,                                             \
    (basename, superclass))

// Specialisations for nodes that declare no user fields (the trait list
// contains only the base expr_id/type entries). The empty case skips the
// chain and forces only the two terminating instantiations directly.
#define ESBMC_INSTANTIATE_TYPE_EMPTY(basename, superclass)                     \
  template class esbmct::type_methods2<                                        \
    BOOST_PP_CAT(basename, 2t),                                                \
    superclass,                                                                \
    typename superclass::traits>;                                              \
  template class esbmct::irep_methods2<                                        \
    BOOST_PP_CAT(basename, 2t),                                                \
    superclass,                                                                \
    typename superclass::traits>

#define ESBMC_INSTANTIATE_EXPR_EMPTY(basename, superclass)                     \
  template class esbmct::expr_methods2<                                        \
    BOOST_PP_CAT(basename, 2t),                                                \
    superclass,                                                                \
    superclass::traits>;                                                       \
  template class esbmct::irep_methods2<                                        \
    BOOST_PP_CAT(basename, 2t),                                                \
    superclass,                                                                \
    superclass::traits>;                                                       \
  template class esbmct::irep_methods2<                                        \
    BOOST_PP_CAT(basename, 2t),                                                \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mp11::mp_pop_front<typename superclass::traits::fields>>
