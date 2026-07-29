#include <util/base/filesystem.h>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <fstream>
#include <vector>

#ifndef _WIN32
#  include <csignal>
#  include <sys/types.h>
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

  std::string contents(static_cast<size_t>(in.tellg()), '\0');
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

  std::ofstream(path).write(s, n);
}
