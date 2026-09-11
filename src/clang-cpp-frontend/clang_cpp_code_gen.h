#ifndef ESBMC_CLANG_CPP_FRONTEND_CLANG_CPP_CODE_GEN_H
#define ESBMC_CLANG_CPP_FRONTEND_CLANG_CPP_CODE_GEN_H

#include <util/symtab/context.h>
#include <util/symtab/symbol.h>

/// Write each class's vtable pointer at the entry of a constructor or
/// destructor body, so a virtual call during construction resolves to that
/// class's overrides. Shared rather than a member of clang_cpp_adjust because
/// the IREP2 C++ pass needs the same code generated and the function reads
/// nothing but the context (docs/roadmap/scope-clang-cpp-irep2.md §3.6).
void gen_vptr_initializations(contextt &context, symbolt &symbol);

#endif
