#pragma once

#include <util/base/compiler_defs.h>
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <llvm/Support/VirtualFileSystem.h>
CC_DIAGNOSTIC_POP()

/**
 * @brief A clang filesystem serving ESBMC's bundled files, overlaid on the
 *        real one.
 *
 * Headers and operational models bundled by scripts/flail.py are served
 * straight from the binary's .rodata: clang reads them without their ever
 * being written to disk. Anything else falls through to the real filesystem,
 * so a user's sources and system headers resolve as usual.
 */
llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> esbmc_clang_vfs();
