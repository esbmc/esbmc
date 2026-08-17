#include <util/base/filesystem.h>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
#  include <io.h>
#else
#  include <csignal>
#  include <sys/file.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

using namespace file_operations;

/* systemd-tmpfiles ages /tmp by wall-clock time, so a verification run that
 * outlives the configured age -- or merely spans a suspend or a clock jump --
 * can have its temporaries unlinked from under it. It takes a shared BSD lock
 * on each directory it descends into and skips anything already locked, along
 * with everything below it, which applications are explicitly invited to use to
 * exclude a subtree (systemd.io/TEMPORARY_DIRECTORIES). Hold such a lock for as
 * long as we own the path. Best effort: filesystems that do not implement
 * flock() simply leave us unlocked, exactly as before. */
static int lock_tmp_path([[maybe_unused]] const std::string &path)
{
#ifdef _WIN32
  return -1;
#else
  int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return -1;

  if (flock(fd, LOCK_SH | LOCK_NB))
  {
    close(fd);
    return -1;
  }
  return fd;
#endif
}

static void unlock_tmp_path(int &fd)
{
#ifndef _WIN32
  if (fd >= 0)
    close(fd); /* releases the lock */
#endif
  fd = -1;
}

static std::vector<std::string> registered_tmp_paths;
static std::vector<long> registered_pgroups;

/* Signal-safe mirrors of the two registries above. A handler may run with the
 * allocator lock held by the code it interrupted, so it must not touch the
 * std:: containers: iterating them races with a concurrent push_back, and
 * clearing the path vector frees the strings and re-enters malloc, which glibc
 * catches as a heap-consistency assertion (#6201). These are fixed-capacity,
 * written only at registration, and read by the handler with plain loads. */
#ifndef _WIN32
static constexpr size_t sig_max_tmps = 32;
static constexpr size_t sig_path_max = 4096;
static char sig_tmp_paths[sig_max_tmps][sig_path_max];
static volatile sig_atomic_t sig_tmp_count = 0;

static constexpr size_t sig_max_pgroups = 64;
static volatile sig_atomic_t sig_pgroups[sig_max_pgroups];
static volatile sig_atomic_t sig_pgroup_count = 0;
#endif

void file_operations::register_tmp_for_cleanup(const std::string &path)
{
  registered_tmp_paths.push_back(path);
#ifndef _WIN32
  size_t n = static_cast<size_t>(sig_tmp_count);
  if (n < sig_max_tmps && path.size() < sig_path_max)
  {
    memcpy(sig_tmp_paths[n], path.c_str(), path.size() + 1);
    sig_tmp_count = static_cast<sig_atomic_t>(n + 1);
  }
#endif
}

void file_operations::cleanup_registered_tmps()
{
  for (const auto &p : registered_tmp_paths)
  {
    boost::system::error_code ec;
    boost::filesystem::remove_all(p, ec);
  }
  registered_tmp_paths.clear();
#ifndef _WIN32
  sig_tmp_count = 0;
#endif
}

void file_operations::register_pgroup_for_cleanup(long pgid)
{
  registered_pgroups.push_back(pgid);
#ifndef _WIN32
  size_t n = static_cast<size_t>(sig_pgroup_count);
  if (n < sig_max_pgroups)
  {
    sig_pgroups[n] = static_cast<sig_atomic_t>(pgid);
    sig_pgroup_count = static_cast<sig_atomic_t>(n + 1);
  }
#endif
}

void file_operations::unregister_pgroup(long pgid)
{
  auto &v = registered_pgroups;
  v.erase(std::remove(v.begin(), v.end(), pgid), v.end());
#ifndef _WIN32
  /* Clearing the slot rather than compacting keeps the mirror append-only, so
   * a handler firing mid-update never observes a shifted or short array. */
  for (sig_atomic_t i = 0; i < sig_pgroup_count; ++i)
    if (sig_pgroups[i] == static_cast<sig_atomic_t>(pgid))
      sig_pgroups[i] = 0;
#endif
}

void file_operations::kill_registered_pgroups()
{
#ifndef _WIN32
  for (long pgid : registered_pgroups)
    if (pgid > 0)
      killpg(static_cast<pid_t>(pgid), SIGKILL);
#endif
  registered_pgroups.clear();
#ifndef _WIN32
  sig_pgroup_count = 0;
#endif
}

#ifndef _WIN32
void file_operations::kill_registered_pgroups_from_signal()
{
  for (sig_atomic_t i = 0; i < sig_pgroup_count; ++i)
  {
    sig_atomic_t pgid = sig_pgroups[i];
    if (pgid > 0)
      killpg(static_cast<pid_t>(pgid), SIGKILL);
  }
}

void file_operations::remove_registered_tmps_from_signal()
{
  /* The temporaries are directory trees, and no async-signal-safe call removes
   * one. fork() is safe, and a child that does nothing but execve() is too, so
   * hand each tree to rm(1): the child never touches the inherited heap.
   * argv is built here from static storage only. Best effort — if rm is
   * missing the tree is simply left for /tmp ageing, as it would have been
   * with no cleanup at all. */
  static char arg_rm[] = "rm";
  static char arg_rf[] = "-rf";
  static char *const envp[] = {nullptr};

  for (sig_atomic_t i = 0; i < sig_tmp_count; ++i)
  {
    if (sig_tmp_paths[i][0] == '\0')
      continue;

    pid_t pid = fork();
    if (pid != 0)
      continue;

    /* Leave the caller's process group so a killpg() of it (signal_catcher)
     * cannot take the child down before it has removed anything. */
    setpgid(0, 0);
    char *argv[] = {arg_rm, arg_rf, sig_tmp_paths[i], nullptr};
    execve("/bin/rm", argv, envp);
    _exit(127);
  }
}
#endif

file_data file_data::bundled(const char *data, size_t size)
{
  file_data f;
  f._borrowed = std::string_view(data, size);
  return f;
}

file_data file_data::owned(std::string data)
{
  file_data f;
  f._owned = std::move(data);
  f._bundled = false;
  return f;
}

std::string_view file_data::view() const noexcept
{
  return _bundled ? _borrowed : std::string_view(_owned);
}

size_t file_data::size() const noexcept
{
  return view().size();
}

bool file_data::is_bundled() const noexcept
{
  return _bundled;
}

static bool is_sep(char c)
{
  return c == '/' || c == '\\';
}

bool file_operations::is_bundled_source(std::string_view file)
{
  /* clang_vfs_path() spells the root "C:/esbmc-vfs" on Windows, where a bare
   * "/esbmc-vfs" would not satisfy clang's absolute-path grammar, and clang
   * hands the rest of the path back with native separators. */
  if (file.size() > 1 && file[1] == ':')
    file.remove_prefix(2);

  constexpr std::string_view root = std::string_view(ESBMC_VFS_ROOT).substr(1);
  return file.size() > root.size() + 1 && is_sep(file[0]) &&
         file.compare(1, root.size(), root) == 0 &&
         is_sep(file[root.size() + 1]);
}

filesystemt &filesystemt::get()
{
  static filesystemt instance;
  return instance;
}

void filesystemt::add_bundled(
  const std::string &path,
  const char *data,
  size_t size)
{
  /* bundled_count() keys the overlay cache in esbmc_clang_vfs(), so silently
   * replacing an entry would leave that cache stale -- in release builds too,
   * hence not an assert. */
  if (!_bundled.emplace(path, std::string_view(data, size)).second)
  {
    fprintf(stderr, "ERROR: bundled file registered twice: %s\n", path.c_str());
    abort();
  }
}

std::optional<file_data> filesystemt::read(const std::string &path) const
{
  auto it = _bundled.find(path);
  if (it != _bundled.end())
    return file_data::bundled(it->second.data(), it->second.size());

  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in)
    return {};

  std::streampos end = in.tellg();
  if (end < 0) /* not seekable */
    return {};

  std::string contents(static_cast<size_t>(end), '\0');
  in.seekg(0);
  in.read(contents.data(), contents.size());
  contents.resize(in.gcount());
  return file_data::owned(std::move(contents));
}

bool filesystemt::exists(const std::string &path) const
{
  return _bundled.count(path) || boost::filesystem::exists(path);
}

bool filesystemt::readable(const std::string &path) const
{
  return _bundled.count(path) || std::ifstream(path);
}

size_t filesystemt::bundled_count() const noexcept
{
  return _bundled.size();
}

std::vector<std::string> filesystemt::list(const std::string &prefix) const
{
  std::vector<std::string> paths;
  for_each_under(prefix, [&paths](const std::string &p, std::string_view) {
    paths.push_back(p);
  });
  return paths;
}

const std::string &
filesystemt::materialize(const std::string &prefix, const std::string &format)
{
  auto it = _materialized.find(prefix);
  if (it != _materialized.end())
    return it->second.path();

  it = _materialized.emplace(prefix, create_tmp_dir(format)).first;
  const std::string &dir = it->second.path();
  for_each_under(
    prefix, [&dir, &prefix](const std::string &p, std::string_view data) {
      std::string_view rel = std::string_view(p).substr(prefix.size());
      while (!rel.empty() && rel.front() == '/')
        rel.remove_prefix(1);
      create_path_and_write(
        dir + "/" + std::string(rel), data.data(), data.size());
    });
  return dir;
}

tmp_path::tmp_path(std::string path, bool keep)
  : _path(std::move(path)), _lock_fd(lock_tmp_path(_path)), _keep(keep)
{
  assert(boost::filesystem::exists(_path));
}

tmp_path::tmp_path(tmp_path &&o)
  : _path(std::move(o._path)), _lock_fd(o._lock_fd), _keep(o._keep)
{
  /* Take over the lock rather than re-acquiring it: a second flock() on the
   * same path from this process would silently succeed and leave the original
   * descriptor leaked. */
  o._lock_fd = -1;
  o._keep = true;
}

tmp_path::~tmp_path()
{
  /* Drop the lock whether or not we remove the path: keeping the descriptor
   * open past our ownership would pin the inode and keep excluding it from
   * ageing forever. */
  unlock_tmp_path(_lock_fd);

  if (_keep)
    return;
  // Best-effort cleanup: the path may already be gone. create_tmp_dir() also
  // hands the path to register_tmp_for_cleanup(), so cleanup_registered_tmps()
  // — invoked from the signal handler before exit() runs static/RAII
  // destructors (see signal_catcher.cpp) — can remove it first. remove_all
  // then returns 0, which is a valid "nothing to remove" outcome, not an
  // error. Use the non-throwing form and tolerate a missing path; asserting
  // removed >= 1 here aborted on SIGTERM/SIGINT (e.g. a benchexec timeout).
  boost::system::error_code ec;
  boost::filesystem::remove_all(_path, ec);
}

tmp_path &tmp_path::operator=(tmp_path o)
{
  swap(*this, o);
  return *this;
}

const std::string &tmp_path::path() const noexcept
{
  return _path;
}

tmp_path &tmp_path::keep(bool yes) &noexcept
{
  _keep = yes;
  return *this;
}

tmp_path &&tmp_path::keep(bool yes) &&noexcept
{
  _keep = yes;
  return std::move(*this);
}

tmp_file::tmp_file(FILE *f, tmp_path path) : tmp_path(std::move(path)), _file(f)
{
  assert(f);
}

tmp_file::~tmp_file()
{
  if (_keep)
    return;
  if (fclose(_file))
    fprintf(
      stderr, "ERROR: temp-file %s: %s\n", path().c_str(), strerror(errno));
}

tmp_file &tmp_file::operator=(tmp_file o)
{
  swap(*this, o);
  return *this;
}

FILE *tmp_file::file() noexcept
{
  return _file;
}

/* unique_path() only invents a name; it does not stake a claim on it. Opening
 * that name with fopen() would happily follow a symlink planted there in the
 * meantime and truncate whatever it points at (CWE-377). O_EXCL fails instead,
 * which makes the caller's loop pick a new name. 0600 also stops the temporary
 * from being world-readable, which fopen()'s 0666 & ~umask allowed. */
static FILE *fopen_exclusive(const std::string &path, const char *mode)
{
#ifdef _WIN32
  int fd =
    _open(path.c_str(), _O_CREAT | _O_EXCL | _O_RDWR, _S_IREAD | _S_IWRITE);
#else
  int fd = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
#endif
  if (fd < 0)
    return NULL;

#ifdef _WIN32
  FILE *f = _fdopen(fd, mode);
  if (!f)
    _close(fd);
#else
  FILE *f = fdopen(fd, mode);
  if (!f)
    close(fd);
#endif
  return f;
}

template <typename F>
static inline std::string with_unique_tmp_path(F &&f, const std::string &format)
{
  using namespace boost::filesystem;
  for (path pattern = temp_directory_path() / format;;)
  {
    path p = unique_path(pattern);
    if (f(p))
      return p.string();
  }
}

tmp_file
file_operations::create_tmp_file(const std::string &format, const char *mode)
{
  FILE *r = NULL;
  std::string path = with_unique_tmp_path(
    [&r, mode](auto path) {
      r = fopen_exclusive(path.string(), mode);
      return r;
    },
    format);
  return {r, {std::move(path)}};
}

tmp_path file_operations::create_tmp_dir(const std::string &format)
{
  std::string dir = with_unique_tmp_path(
    [](auto path) { return boost::filesystem::create_directory(path); },
    format);
  register_tmp_for_cleanup(dir);
  return {std::move(dir)};
}

const std::string
file_operations::get_unique_tmp_path(const std::string &format)
{
  // create_directory() reports whether *this* call created the directory, so
  // testing its result claims the name atomically. Testing exists() first and
  // creating afterwards left a window in which another process could take the
  // name between the two calls.
  return with_unique_tmp_path(
    [](const auto &path) { return boost::filesystem::create_directory(path); },
    format);
}

void file_operations::create_path_and_write(
  const std::string &path,
  const char *s,
  size_t n)
{
  boost::filesystem::path p(path);
  if (!boost::filesystem::exists(p.parent_path()))
    boost::filesystem::create_directories(p.parent_path());

  std::ofstream(path).write(s, n);
}
