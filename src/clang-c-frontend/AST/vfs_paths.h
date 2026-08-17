#pragma once

#include <string>
#include <util/base/filesystem.h>

/**
 * @brief A bundled path spelled the way clang's path grammar requires.
 *
 * Identity on POSIX. On Windows llvm::sys::path::is_absolute() also wants a
 * root name, so a bare "/esbmc-vfs/..." would count as relative and header
 * search would never reach the overlay. The drive letter satisfies that
 * grammar only; the overlay sits above the real filesystem, so a real
 * C:\esbmc-vfs is shadowed rather than consulted.
 */
inline std::string clang_vfs_path(const std::string &bundled)
{
#ifdef _WIN32
  return "C:" + bundled;
#else
  return bundled;
#endif
}

inline std::string clang_vfs_root()
{
  return clang_vfs_path(file_operations::ESBMC_VFS_ROOT);
}
