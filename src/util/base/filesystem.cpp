#include <util/base/filesystem.h>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <cassert>
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

void file_operations::register_tmp_for_cleanup(const std::string &path)
{
  registered_tmp_paths.push_back(path);
}

void file_operations::cleanup_registered_tmps()
{
  for (const auto &p : registered_tmp_paths)
  {
    boost::system::error_code ec;
    boost::filesystem::remove_all(p, ec);
  }
  registered_tmp_paths.clear();
}

void file_operations::register_pgroup_for_cleanup(long pgid)
{
  registered_pgroups.push_back(pgid);
}

void file_operations::unregister_pgroup(long pgid)
{
  auto &v = registered_pgroups;
  v.erase(std::remove(v.begin(), v.end(), pgid), v.end());
}

void file_operations::kill_registered_pgroups()
{
#ifndef _WIN32
  for (long pgid : registered_pgroups)
    if (pgid > 0)
      killpg(static_cast<pid_t>(pgid), SIGKILL);
#endif
  registered_pgroups.clear();
}

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

/* Whether `want` occupies whole components of `path` starting at `at`. Both
 * separators are accepted: clang hands VFS paths back native on Windows. */
static bool
components_match(std::string_view path, size_t at, std::string_view want)
{
  if (path.size() - at < want.size())
    return false;
  for (size_t i = 0; i < want.size(); ++i)
    if (path[at + i] != want[i] && !(is_sep(path[at + i]) && want[i] == '/'))
      return false;
  size_t end = at + want.size();
  return end == path.size() || is_sep(path[end]);
}

bool file_operations::is_bundled_source(std::string_view file)
{
  /* clang_vfs_path() spells the root "C:/esbmc-vfs" on Windows, where a bare
   * "/esbmc-vfs" would not satisfy clang's absolute-path grammar. */
  std::string_view rooted = file;
  if (rooted.size() > 1 && rooted[1] == ':')
    rooted.remove_prefix(2);
  if (components_match(rooted, 0, ESBMC_VFS_ROOT))
    return true;

  /* The c2goto library arrives the other way round: it is compiled into the
   * goto binary at build time, so its symbols carry whatever absolute path the
   * build tree had. With no prefix to anchor against, require the component
   * sequence -- only an ESBMC checkout has one. */
  for (size_t at = 0; at < file.size(); ++at)
    if (
      (at == 0 || is_sep(file[at - 1])) &&
      components_match(file, at, "src/c2goto/library"))
      return true;
  return false;
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
   * replacing an entry would leave that cache stale. */
  assert(!_bundled.count(path));
  _bundled[path] = std::string_view(data, size);
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
