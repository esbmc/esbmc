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
/**
 * @brief Root of the bundled path namespace, e.g. /esbmc-vfs/clang/include.
 *
 * A path rather than a URI scheme, because consumers that resolve these
 * strings understand path syntax and nothing else. Reserved, so it never
 * names anything on the real filesystem.
 */
inline constexpr const char *ESBMC_VFS_ROOT = "/esbmc-vfs";

/**
 * @brief Read-only contents of a file, either bundled into the ESBMC binary
 *        or read from disk.
 *
 * Files bundled by scripts/flail.py are already resident in the binary's
 * .rodata, so view() borrows them directly and reading one costs no
 * allocation. Contents read from disk are held in an internal buffer that
 * view() borrows from instead. Either way the view stays valid for as long as
 * the file_data does.
 *
 * view() is null-terminated at view().size(), which clang's Lexer requires of
 * any buffer handed to it directly. std::string provides that for owned
 * contents; for bundled ones it comes from the NUL sentinel flail appends past
 * the end of each array. Copying is disabled so that contents read from disk,
 * which may be large, are never duplicated by accident.
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
 * This function does not have guarantee that will finish
 * and can be run forever until it sees an available spot.
 *
 * @param format A string in the file specification
 */
const std::string get_unique_tmp_path(const std::string &format);

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

/**
 * @brief ESBMC's filesystem: files bundled into the binary, overlaid on the
 *        real one.
 *
 * Every file scripts/flail.py bundles is registered here at startup under a
 * path below ESBMC_VFS_ROOT, costing nothing but a map entry -- the bytes stay
 * in .rodata. Lookups resolve by path: the registry first, the real filesystem
 * on a miss. Callers therefore need not know, and cannot tell, which of the
 * two answered.
 *
 * Bundled files still have to be written out for anything ESBMC cannot reach
 * into: a forked python3 or solc has its own address space and can only read
 * real files. materialize() serves those.
 *
 * Registration is unsynchronised and must complete before any concurrent
 * read; calling the modules' register_bundled() from main() satisfies that.
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
      f(it->first, it->second);
  }

public:
  static filesystemt &get();

  /** @brief Registers static file contents; see file_data::bundled(). */
  void add_bundled(const std::string &path, const char *data, size_t size);

  /** @brief Contents of `path`, or nothing if it is neither bundled nor a
   *         readable file. */
  std::optional<file_data> read(const std::string &path) const;

  bool exists(const std::string &path) const;

  /** @brief How many bundled files are registered. Only ever grows, so a
   *         change means new registrations arrived. */
  size_t bundled_count() const noexcept;

  /**
   * @brief Every bundled path below `prefix`, at any depth.
   *
   * Paths are keys of a flat map, so directories exist only as structure
   * within them and the walk is inherently recursive.
   */
  std::vector<std::string> list(const std::string &prefix) const;

  /**
   * @brief Writes every bundled file below `prefix` into a fresh temporary
   *        directory named after `format`, and returns that directory.
   *
   * The result is cached per prefix, so a subtree is written at most once per
   * run and later calls just return the path. The directory is removed when
   * ESBMC exits.
   */
  const std::string &
  materialize(const std::string &prefix, const std::string &format);
};
} // namespace file_operations
