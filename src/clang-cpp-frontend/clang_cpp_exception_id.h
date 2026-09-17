#ifndef ESBMC_CLANG_CPP_FRONTEND_CLANG_CPP_EXCEPTION_ID_H
#define ESBMC_CLANG_CPP_FRONTEND_CLANG_CPP_EXCEPTION_ID_H

#include <util/irep/std_types.h>
#include <util/symtab/namespace.h>
#include <string>
#include <vector>

/// The catchable-type ids a thrown or caught type resolves to, most derived
/// first. Shared rather than a member of clang_cpp_adjust because the IREP2 C++
/// pass needs the same answer and the function reads nothing but the namespace
/// (docs/roadmap/scope-clang-cpp-irep2.md §3.3).
void convert_exception_id(
  const namespacet &ns,
  const typet &type,
  const std::string &suffix,
  std::vector<irep_idt> &ids,
  bool is_catch = false);

/// Resolve a function type's dynamic exception specification. The converter
/// stashes the declared types of a `throw(T...)` under "exception_spec_decl";
/// this turns them into exception ids and stores the list under
/// exception_specificationt's types_attribute(). A no-op for any other kind of
/// specification.
///
/// Shared for the same reason convert_exception_id is: the IREP2 C++ pass has
/// to reach the same answer, and this reads nothing but the namespace. Left
/// unresolved, the spec permits nothing and every throw through such a function
/// reports "exception specification violated"
/// (docs/roadmap/scope-clang-cpp-irep2.md §7.7).
void finalize_exception_specification(const namespacet &ns, typet &type);

#endif
