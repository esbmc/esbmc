#include <util/base/compiler_defs.h>
CC_DIAGNOSTIC_PUSH()
CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()
#include <llvm/Support/MemoryBuffer.h>
CC_DIAGNOSTIC_POP()

#include <clang-c-frontend/AST/vfs_adapter.h>
#include <util/base/filesystem.h>

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> esbmc_clang_vfs()
{
  static llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> cached;
  static size_t cached_count = 0;

  /* Frontends register their bundled files lazily, so the overlay is rebuilt
   * whenever more have arrived since it was last built. Without this a
   * translation unit parsed before, say, the libc registers would see a stale
   * filesystem. */
  auto &fs = file_operations::filesystemt::get();
  if (cached && cached_count == fs.bundled_count())
    return cached;

  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> in_memory(
    new llvm::vfs::InMemoryFileSystem);

  for (const std::string &path : fs.list(file_operations::ESBMC_VFS_ROOT))
  {
    std::optional<file_operations::file_data> contents = fs.read(path);
    /* Everything under the VFS root is bundled, so the bytes live in .rodata:
     * getMemBuffer borrows them rather than copying, and the buffer stays
     * valid after `contents` goes out of scope. A non-bundled entry would not
     * survive that, hence the guard. */
    if (!contents || !contents->is_bundled())
      continue;

    in_memory->addFile(
      path,
      0,
      llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(contents->view().data(), contents->size()),
        path,
        /*RequiresNullTerminator=*/true));
  }

  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlay(
    new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
  overlay->pushOverlay(in_memory);

  cached = overlay;
  cached_count = fs.bundled_count();
  return cached;
}
