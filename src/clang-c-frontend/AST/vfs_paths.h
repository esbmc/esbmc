#pragma once

#include <string>
#include <util/base/filesystem.h>

/**
 * @brief A bundled path spelled the way clang's path grammar requires.
 *
 * Identity on POSIX. On Windows llvm::sys::path::is_absolute() demands a root
 * name as well as a root directory, so a bare "/esbmc-vfs/..." is not absolute
 * there and clang's header search would treat it as relative and never reach
 * the in-memory overlay. The drive letter is a synthetic marker satisfying
 * that grammar, not a claim about storage: these paths resolve to memory, and
 * the overlay sits above the real filesystem, so a real C:\esbmc-vfs would be
 * shadowed rather than consulted.
 *
 * Header-only and free of LLVM includes so the frontends that build clang
 * arguments can use it without taking on a dependency.
 */
inline std::string clang_vfs_path(const std::string &bundled)
{
#ifdef _WIN32
  return "C:" + bundled;
#else
  return bundled;
#endif
}

/** @brief clang_vfs_path() applied to the bundled root. */
inline std::string clang_vfs_root()
{
  return clang_vfs_path(file_operations::ESBMC_VFS_ROOT);
}
