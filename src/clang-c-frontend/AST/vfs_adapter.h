#pragma once

#include <util/base/compiler_defs.h>
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <llvm/Support/VirtualFileSystem.h>
CC_DIAGNOSTIC_POP()

/**
 * @brief A clang filesystem serving ESBMC's bundled files straight from
 *        .rodata, overlaid on the real one.
 *
 * Anything not bundled falls through, so user sources and system headers
 * resolve as usual.
 */
llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> esbmc_clang_vfs();
