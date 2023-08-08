#pragma once
#include <irep2/irep2_meta_templates.h>

/********************** Constants and explicit instantiations *****************/

// This has become particularly un-fun with the arrival of gcc 6.x and clang
// 3.8 (roughly). Both are very aggressive wrt. whether templates are actually
// instantiated or not, and refuse to instantiate template base classes
// implicitly. Unfortunately, we relied on that before; it might still be
// instantiating some of the irep_methods2 chain, but it's not doing the
// method definitions, which leads to linking failures later.
//
// I've experimented with a variety of ways to implicitly require each method
// of our template chain, but none seem to succeed, and the compiler goes a
// long way out of it's path to avoid these instantiations. The real real
// issue seems to be virtual functions, the compiler can jump through many
// hoops to get method addresses out of the vtable, rather than having to
// implicitly define it. One potential workaround may be a non-virtual method
// that gets defined that calls all virtual methods explicitly?
//
// Anyway: the workaround is to explicitly instantiate each level of the
// irep_methods2 hierarchy, with associated pain and suffering. This means
// that our template system isn't completely variadic, you have to know how
// many levels to instantiate when you reach this level, explicitly. Which
// sucks, but is a small price to pay.

#undef irep_typedefs
#undef irep_typedefs_empty

#define irep_typedefs0(basename, superclass)                                   \
  template class esbmct::                                                      \
    irep_methods2<basename##2t, superclass, superclass::traits>;               \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;

#define irep_typedefs1(basename, superclass)                                   \
  template class esbmct::                                                      \
    irep_methods2<basename##2t, superclass, superclass::traits>;               \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;         \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename superclass::traits::fields>::type>::type>;

#define irep_typedefs2(basename, superclass)                                   \
  template class esbmct::                                                      \
    irep_methods2<basename##2t, superclass, superclass::traits>;               \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;         \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename superclass::traits::fields>::type>::type>;                      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<                                                     \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>;

#define irep_typedefs3(basename, superclass)                                   \
  template class esbmct::                                                      \
    irep_methods2<basename##2t, superclass, superclass::traits>;               \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;         \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename superclass::traits::fields>::type>::type>;                      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<                                                     \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>;             \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>::type>;

#define irep_typedefs4(basename, superclass)                                   \
  template class esbmct::                                                      \
    irep_methods2<basename##2t, superclass, superclass::traits>;               \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;         \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename superclass::traits::fields>::type>::type>;                      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<                                                     \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>;             \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>::type>;      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename boost::mpl::pop_front<typename superclass::traits::fields>::  \
          type>::type>::type>::type>::type>;

#define irep_typedefs5(basename, superclass)                                   \
  template class esbmct::                                                      \
    irep_methods2<basename##2t, superclass, superclass::traits>;               \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;         \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename superclass::traits::fields>::type>::type>;                      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<                                                     \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>;             \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>::type>;      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename boost::mpl::pop_front<typename superclass::traits::fields>::  \
          type>::type>::type>::type>::type>;                                   \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<boost::mpl::pop_front<                               \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename boost::mpl::pop_front<typename boost::mpl::pop_front<         \
          typename superclass::traits::fields>::type>::type>::type>::type>::   \
                            type>::type>;

#define irep_typedefs6(basename, superclass)                                   \
  template class esbmct::                                                      \
    irep_methods2<basename##2t, superclass, superclass::traits>;               \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename superclass::traits::fields>::type>;         \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename superclass::traits::fields>::type>::type>;                      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<                                                     \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>;             \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename superclass::traits::fields>::type>::type>::type>::type>;      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<typename boost::mpl::pop_front<                      \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename boost::mpl::pop_front<typename superclass::traits::fields>::  \
          type>::type>::type>::type>::type>;                                   \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<boost::mpl::pop_front<                               \
      typename boost::mpl::pop_front<typename boost::mpl::pop_front<           \
        typename boost::mpl::pop_front<typename boost::mpl::pop_front<         \
          typename superclass::traits::fields>::type>::type>::type>::type>::   \
                            type>::type>;                                      \
  template class esbmct::irep_methods2<                                        \
    basename##2t,                                                              \
    superclass,                                                                \
    superclass::traits,                                                        \
    boost::mpl::pop_front<                                                     \
      typename boost::mpl::pop_front<boost::mpl::pop_front<                    \
        typename boost::mpl::pop_front<typename boost::mpl::pop_front<         \
          typename boost::mpl::pop_front<typename boost::mpl::pop_front<       \
            typename superclass::traits::fields>::type>::type>::type>::type>:: \
                                       type>::type>::type>;

////////////////////////////

#define type_typedefs1(basename, superclass)                                   \
  template class esbmct::                                                      \
    type_methods2<basename##2t, superclass, typename superclass::traits>;      \
  irep_typedefs1(basename, superclass)

#define type_typedefs2(basename, superclass)                                   \
  template class esbmct::                                                      \
    type_methods2<basename##2t, superclass, typename superclass::traits>;      \
  irep_typedefs2(basename, superclass)

#define type_typedefs3(basename, superclass)                                   \
  template class esbmct::                                                      \
    type_methods2<basename##2t, superclass, typename superclass::traits>;      \
  irep_typedefs3(basename, superclass)

#define type_typedefs4(basename, superclass)                                   \
  template class esbmct::                                                      \
    type_methods2<basename##2t, superclass, typename superclass::traits>;      \
  irep_typedefs4(basename, superclass)

#define type_typedefs5(basename, superclass)                                   \
  template class esbmct::                                                      \
    type_methods2<basename##2t, superclass, typename superclass::traits>;      \
  irep_typedefs5(basename, superclass)

#define type_typedefs6(basename, superclass)                                   \
  template class esbmct::                                                      \
    type_methods2<basename##2t, superclass, typename superclass::traits>;      \
  irep_typedefs6(basename, superclass)

#define type_typedefs_empty(basename, superclass)                              \
  template class esbmct::                                                      \
    type_methods2<basename##2t, superclass, typename superclass::traits>;      \
  template class esbmct::                                                      \
    irep_methods2<basename##2t, superclass, typename superclass::traits>;
