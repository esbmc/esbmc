#pragma once

#include <cstdio> /* FILE */
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief this file will contains helper functions for manipulating
 *        files
 */

namespace file_operations
{
/** @brief Root of the bundled path namespace, e.g. /esbmc-vfs/clang/include.
 *         Reserved, so it never names anything on the real filesystem. */
inline constexpr const char *ESBMC_VFS_ROOT = "/esbmc-vfs";

/** @brief True if @p file names one of ESBMC's own operational-model or
 *         library sources rather than user code. */
bool is_bundled_source(std::string_view file);

/**
 * @brief Read-only contents of a file, either bundled into the ESBMC binary
 *        or read from disk.
 *
 * Bundled contents are borrowed from .rodata, so reading one costs no
 * allocation; contents read from disk are owned. Either way view() is
 * NUL-terminated at size(), which clang's Lexer requires of buffers handed to
 * it directly -- for bundled files that comes from the sentinel flail appends.
 */
class file_data
{
  std::string _owned;
  std::string_view _borrowed;
  bool _bundled = true;

public:
  file_data() = default;
  file_data(const file_data &) = delete;
  file_data &operator=(const file_data &) = delete;
  file_data(file_data &&) = default;
  file_data &operator=(file_data &&) = default;

  /**
   * @brief Borrows `size` bytes of static storage without copying them.
   *
   * `data[size]` must be `'\0'`; pass the `_size` symbol flail generates,
   * which excludes the sentinel it appends.
   */
  static file_data bundled(const char *data, size_t size);

  /** @brief Takes ownership of contents read from disk. */
  static file_data owned(std::string data);

  std::string_view view() const noexcept;
  size_t size() const noexcept;
  bool is_bundled() const noexcept;
};

/**
 * @brief Represents a temporary path, which is (optionally) removed by the
 *        destructor.
 *
 * On destruction, optionally (default: yes), the path removed along with all
 * contained paths if it points to a directory. The default ctor is provided
 * only to ease array allocation; it does not construct valid temporary paths.
 * As an instance of this class represents a bound resource, it cannot be
 * copied, only moved.
 */
class tmp_path
{
  std::string _path;
  /** Descriptor holding a BSD lock on the path; -1 when unlocked. Held for as
   *  long as this object owns the path, so systemd-tmpfiles skips it. */
  int _lock_fd = -1;

protected:
  bool _keep = true;

public:
  tmp_path() = default;
  tmp_path(std::string path, bool keep = false);
  tmp_path(const tmp_path &) = delete;
  tmp_path(tmp_path &&o);

  ~tmp_path();

  tmp_path &operator=(tmp_path o);

  friend void swap(tmp_path &a, tmp_path &b)
  {
    using std::swap;
    swap(a._path, b._path);
    swap(a._keep, b._keep);
    swap(a._lock_fd, b._lock_fd);
  }

  const std::string &path() const noexcept;

  tmp_path &keep(bool yes) &noexcept;
  tmp_path &&keep(bool yes) &&noexcept;
};

/**
 * @brief Temporary path to an open file with an associated `FILE` handle.
 *
 * On destruction, optionally (default: yes), the file is closed and the path
 * removed. The default ctor is provided only to ease array allocation; it
 * does not construct valid temporary files. As an instance of this class
 * represents a bound resource, it cannot be copied, only moved.
 */
class tmp_file : public tmp_path
{
  FILE *_file;

public:
  tmp_file() = default;
  tmp_file(FILE *f, tmp_path path);
  tmp_file(const tmp_file &) = delete;
  tmp_file(tmp_file &&o) = default;

  ~tmp_file();

  tmp_file &operator=(tmp_file o);

  friend void swap(tmp_file &a, tmp_file &b)
  {
    using std::swap;
    swap(static_cast<tmp_path &>(a), static_cast<tmp_path &>(b));
    swap(a._file, b._file);
  }

  FILE *file() noexcept;
};

/**
 * @brief Generates a unique path based on the format
 *
 * In Linux, running this function with "esbmc-%%%%" will
 * return a string such as "/tmp/esbmc-0001" or "/tmp/esbmc-8787".
 *
 * The directory is created before the path is returned, so the name is the
 * caller's for as long as it keeps it. Unlike create_tmp_dir() the result is
 * not registered for cleanup: the caller owns removal.
 *
 * This function does not have guarantee that will finish
 * and can be run forever until it sees an available spot.
 *
 * @param format A string in the file specification
 */
const std::string get_unique_tmp_path(const std::string &format);

/** The file is created exclusively and with 0600 permissions, so it is never
 *  a pre-existing path (or a symlink to one) that this would clobber. */
tmp_file create_tmp_file(
  const std::string &format = "esbmc.%%%%-%%%%-%%%%",
  const char *mode = "w+");

tmp_path create_tmp_dir(const std::string &format = "esbmc.%%%%-%%%%-%%%%");

/**
 *  @brief Creates all folders needed for a path
 *
 * std::ofstream will not create folders needed for a
 * complete path. This will generate the folder and the file
 * contents
 */
void create_path_and_write(const std::string &path, const char *s, size_t n);

/**
 * @brief The register_*() registries below are process-global, append-only and
 *        unsynchronised; each is read by the function named alongside it. A
 *        signal landing mid-append can observe a half-written registry.
 */

/** @brief Temporary paths, read by cleanup_registered_tmps() from the signal
 *         handlers, which run before exit() reaches any destructor. */
void register_tmp_for_cleanup(const std::string &path);
void cleanup_registered_tmps();

/**
 * @brief Track child process groups so the signal/timeout exit paths can
 * kill them.
 *
 * A backend that spawns an external solver into its own process group (so
 * the group can be killed as a unit, MPI ranks included) registers the pgid
 * here. On a timeout or fatal signal ESBMC exits without running
 * destructors; kill_registered_pgroups(), called from those handlers, sends
 * SIGKILL to each still-registered group so the children do not linger.
 * unregister_pgroup() is called once the child has been reaped normally.
 * No-op on Windows. `pgid` is a pid_t widened to long to keep this header
 * POSIX-free.
 */
void register_pgroup_for_cleanup(long pgid);
void unregister_pgroup(long pgid);
void kill_registered_pgroups();

#ifndef _WIN32
/**
 * @brief Async-signal-safe counterparts of the two cleanup calls above, for
 * use from a signal handler.
 *
 * The ordinary versions walk std:: containers and, for the temporaries, run
 * boost::filesystem::remove_all; both allocate, so a handler interrupting the
 * allocator deadlocks or trips glibc's heap assertion (#6201). These read
 * fixed-capacity mirrors populated at registration time and call nothing
 * outside POSIX's async-signal-safe set. Neither clears the mirror: a handler
 * runs once, on the way to _exit().
 */
void kill_registered_pgroups_from_signal();
void remove_registered_tmps_from_signal();
#endif

/**
 * @brief Files bundled into the binary, overlaid on the real filesystem.
 *
 * Files scripts/flail.py bundles are registered under ESBMC_VFS_ROOT. read()
 * checks the registry first and falls back to disk, so callers cannot tell
 * which layer answered. materialize() writes a subtree out for consumers that
 * cannot read ESBMC's memory: a forked python3 or solc.
 */
class filesystemt
{
  std::map<std::string, std::string_view> _bundled;
  std::map<std::string, tmp_path> _materialized;

  template <typename F>
  void for_each_under(const std::string &prefix, F &&f) const
  {
    for (auto it = _bundled.lower_bound(prefix);
         it != _bundled.end() &&
         it->first.compare(0, prefix.size(), prefix) == 0;
         ++it)
      /* Whole components only: "/x/libc" must not select "/x/libcxx/y". */
      if (it->first.size() == prefix.size() || it->first[prefix.size()] == '/')
        f(it->first, it->second);
  }

public:
  static filesystemt &get();

  /** @brief Registers static file contents; see file_data::bundled().
   *         `path` must not already be registered. */
  void add_bundled(const std::string &path, const char *data, size_t size);

  /** @brief Contents of `path`, or nothing if it is neither bundled nor a
   *         readable file. */
  std::optional<file_data> read(const std::string &path) const;

  bool exists(const std::string &path) const;

  /** @brief Whether the contents of `path` can actually be obtained. Unlike
   *         exists(), a file that refuses to open does not qualify. */
  bool readable(const std::string &path) const;

  /** @brief How many bundled files are registered. Only ever grows, so a
   *         change means new registrations arrived. */
  size_t bundled_count() const noexcept;

  /** @brief Every bundled path below `prefix`, at any depth. */
  std::vector<std::string> list(const std::string &prefix) const;

  /**
   * @brief Writes every bundled file below `prefix` into a fresh temporary
   *        directory named after `format`, and returns that directory.
   *
   * Cached per prefix, so a subtree is written at most once per run. The
   * directory is removed when ESBMC exits.
   */
  const std::string &
  materialize(const std::string &prefix, const std::string &format);
};
} // namespace file_operations
