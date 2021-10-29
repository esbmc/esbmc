#include <util/filesystem.h>
#include <boost/filesystem.hpp>

using namespace file_operations;

tmp_dir::tmp_dir(std::string path, bool keep)
  : _path(std::move(path)), _keep(keep)
{
  assert(boost::filesystem::is_directory(_path));
}

tmp_dir::~tmp_dir()
{
  if(_keep)
    return;
  [[maybe_unused]] uintmax_t removed = boost::filesystem::remove_all({_path});
  assert(removed >= 1 && "expected to remove temp dir");
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
  } while(
    boost::filesystem::exists(path)); // TODO: This may cause infinite loop

  // If path folders doesn't exist, create then
  boost::filesystem::create_directories(path);
  if(!boost::filesystem::is_directory(path))
  {
    assert(!"Can't create temporary directory");
    abort();
  }

  return path.string();
}
