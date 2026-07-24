#include <util/filesystem.h>
#include <util/picosha2.h>
#include <ac_config.h>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <vector>

#ifndef _WIN32
#  include <csignal>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

using namespace file_operations;

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

tmp_path::tmp_path(std::string path, bool keep)
  : _path(std::move(path)), _keep(keep)
{
  assert(boost::filesystem::exists(_path));
}

tmp_path::tmp_path(tmp_path &&o) : tmp_path(std::move(o._path), o._keep)
{
  o._keep = true;
}

tmp_path::~tmp_path()
{
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

template <typename F>
static inline std::string with_unique_path(
  const boost::filesystem::path &base,
  const std::string &format,
  F &&f)
{
  using namespace boost::filesystem;
  for (path pattern = base / format;;)
  {
    path p = unique_path(pattern);
    if (f(p))
      return p.string();
  }
}

template <typename F>
static inline std::string with_unique_tmp_path(F &&f, const std::string &format)
{
  return with_unique_path(
    boost::filesystem::temp_directory_path(), format, std::forward<F>(f));
}

tmp_file
file_operations::create_tmp_file(const std::string &format, const char *mode)
{
  FILE *r = NULL;
  std::string path = with_unique_tmp_path(
    [&r, mode](auto path) {
      r = fopen(path.string().c_str(), mode);
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
  // Get the temp file dir
  const boost::filesystem::path tmp_path =
    boost::filesystem::temp_directory_path();

  // Define the pattern for the name
  const std::string pattern = (tmp_path / format.c_str()).string();
  boost::filesystem::path path;

  // Try to get a name that is not used already e.g. esbmc.0000-0000
  do
  {
    path = boost::filesystem::unique_path(pattern);
  } while (
    boost::filesystem::exists(path)); // TODO: This may cause infinite loop

  // If path folders doesn't exist, create then
  boost::filesystem::create_directories(path);
  if (!boost::filesystem::is_directory(path))
  {
    assert(!"Can't create temporary directory");
    abort();
  }

  return path.string();
}

void file_operations::create_path_and_write(
  const std::string &path,
  const char *s,
  size_t n)
{
  boost::filesystem::path p(path);
  if (!boost::filesystem::exists(p.parent_path()))
    boost::filesystem::create_directories(p.parent_path());

  /* Report a short write (ENOSPC/EIO/EDQUOT) rather than silently truncating.
   * cached_extract_dir() relies on this to avoid publishing a partial entry. */
  std::ofstream f(path);
  f.exceptions(std::ios::failbit | std::ios::badbit);
  f.write(s, n);
}

/* Create `p` (mode 0700) if absent and confirm it is a directory we own.
 *
 * The cache root has a predictable, shared name (`esbmc-cache-<uid>` in a temp
 * dir every user can write to), so on a multi-user machine we can easily meet a
 * directory of that name that is not the one we would have made: a leftover
 * from another user, a stale root from an older build, or one created with the
 * wrong mode. Accepting only a directory we own, and that only we can enter,
 * keeps entries reused across our own runs from being confused with anyone
 * else's. Anything else is declined and the caller falls back to a
 * randomly-named per-run temp dir. */
static bool ensure_private_dir(const boost::filesystem::path &p)
{
#ifndef _WIN32
  if (mkdir(p.c_str(), 0700) && errno != EEXIST)
    return false;

  struct stat st;
  if (lstat(p.c_str(), &st))
    return false;

  /* mkdir() grants ownership to the effective uid, so compare against that. */
  return S_ISDIR(st.st_mode) && st.st_uid == geteuid() && !(st.st_mode & 077);
#else
  /* On Windows the per-user profile and temp directories are already
   * per-account, so plain creation suffices. */
  boost::system::error_code ec;
  boost::filesystem::create_directories(p, ec);
  return boost::filesystem::is_directory(p);
#endif
}

/* Root under which cached entries live, or empty if none is usable.
 *
 * Entries live one level down, inside a root we own at mode 0700. Checking the
 * root once rather than every entry is deliberate: nothing but our own runs can
 * put anything inside a 0700 directory we own, so each entry we find there is
 * one we wrote. That is what lets the "lost the publish race, reuse the existing
 * entry" path below treat an already-present entry as ours without re-checking
 * it. */
static boost::filesystem::path cache_root()
{
  namespace fs = boost::filesystem;
  boost::system::error_code ec;
  fs::path root;

  const char *explicit_dir = getenv("ESBMC_CACHE_DIR");
  if (explicit_dir && *explicit_dir)
    root = fs::path(explicit_dir);
  else
  {
#ifdef ESBMC_CACHE_IN_HOME
#  ifdef _WIN32
    const char *base = getenv("LOCALAPPDATA");
    if (!base || !*base)
      return {};
    root = fs::path(base) / "esbmc" / "cache";
#  else
    /* XDG requires relative base directories to be ignored. */
    const char *xdg = getenv("XDG_CACHE_HOME");
    const char *home = getenv("HOME");
    if (xdg && *xdg == '/')
      root = fs::path(xdg) / "esbmc";
    else if (home && *home == '/')
      root = fs::path(home) / ".cache" / "esbmc";
    else
      return {};
#  endif
#else
    root = fs::temp_directory_path(ec);
    if (ec)
      return {};
#  ifndef _WIN32
    /* The temp dir is shared between users, so the uid in the name gives each
     * user a distinct root by default. It is a naming convention for
     * separation, not a guarantee — ensure_private_dir() confirms we own
     * whatever is actually there. */
    root /= "esbmc-cache-" + std::to_string(getuid());
#  else
    root /= "esbmc-cache";
#  endif
#endif
  }

  fs::create_directories(root.parent_path(), ec);
  if (ensure_private_dir(root))
    return root;

  /* A user who set ESBMC_CACHE_DIR expects it to be used; if it is not a
   * directory we own at 0700 we cannot, so say so rather than silently
   * reverting to per-run extraction. The default root falls back quietly. */
  if (explicit_dir && *explicit_dir)
    fprintf(
      stderr,
      "warning: ESBMC_CACHE_DIR '%s' is not a 0700 directory owned by this "
      "user; extracting internals per-run instead\n",
      explicit_dir);
  return {};
}

tmp_path file_operations::cached_extract_dir(
  const std::string &name,
  const std::string &content_key,
  const std::function<void(const std::string &)> &extract)
{
  namespace fs = boost::filesystem;

  static const fs::path root = cache_root();
  if (!root.empty())
  {
    /* 64 bits of SHA-256 keeps the path short; the key only has to separate
     * the payloads of distinct builds, not resist collisions. */
    fs::path target =
      root /
      (name + "-" + picosha2::hash256_hex_string(content_key).substr(0, 16));
    if (fs::is_directory(target))
      return {target.string(), true};

    /* Extraction writes into a staging dir and only becomes visible at `target`
     * via an atomic rename, so a hit is always a complete entry. If any write
     * fails (ENOSPC/EIO) or the root becomes unusable, we must not leave a
     * partial tree behind for the next run to trust — discard it and fall
     * through to a per-run extraction instead. */
    try
    {
      /* The tmp_path guard removes the staging dir at every exit from this
       * scope; on the publish path that is a no-op, the rename having moved
       * it away already. */
      tmp_path staging(with_unique_path(
        root, "." + name + ".tmp-%%%%-%%%%-%%%%", [](const fs::path &p) {
          return fs::create_directory(p);
        }));
      register_tmp_for_cleanup(staging.path());
      extract(staging.path());

      boost::system::error_code ec;
      fs::rename(staging.path(), target, ec);
      /* We published, or a concurrent ESBMC published the same content first —
       * either way `target` is now a complete entry. */
      if (!ec || fs::is_directory(target))
        return {target.string(), true};
    }
    catch (const std::exception &)
    {
      /* Swallowing is safe because the fallback below re-runs extract without
       * a guard: a transient failure (a full disk that was since freed, a lost
       * race) recovers there, while a deterministic one re-throws and
       * propagates rather than being silently hidden. */
    }
  }

  /* No usable cache root, or extraction/publish failed: extract into a private
   * temp dir removed on exit — the behaviour from before the cache existed. */
  tmp_path dir = create_tmp_dir(name + "-%%%%-%%%%-%%%%");
  extract(dir.path());
  return dir;
}
