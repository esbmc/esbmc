#include <util/filesystem.h>
#include <boost/filesystem.hpp>
#include <fstream>

using namespace file_operations;

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
  uintmax_t removed [[maybe_unused]] = boost::filesystem::remove_all(_path);
  assert(removed >= 1 && "expected to remove temp path");
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
  return {with_unique_tmp_path(
    [](auto path) { return boost::filesystem::create_directory(path); },
    format)};
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
