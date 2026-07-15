#pragma once

#include <cstdio> /* FILE */
#include <string>

/**
 * @brief this file will contains helper functions for manipulating
 *        files
 */

namespace file_operations
{
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
} // namespace file_operations
