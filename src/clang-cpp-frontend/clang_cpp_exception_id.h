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

#endif
