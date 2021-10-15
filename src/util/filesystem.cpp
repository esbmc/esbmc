#include <util/filesystem.h>
#include <boost/filesystem.hpp>

std::string file_operations::get_unique_tmp_path(const char *format)
{
  // Get the temp file dir
  const boost::filesystem::path tmp_path =
    boost::filesystem::temp_directory_path();

  // Define the pattern for the name
  const std::string pattern = (tmp_path / format).string();
  boost::filesystem::path path;

  // Try to get a name that is not used already e.g. esbmc.0000-0000
  do
  {
    path = boost::filesystem::unique_path(pattern);
  } while(
    boost::filesystem::exists(path)); // TODO: This may cause infinite recursion

  // If path folders doesn't exist, create then
  boost::filesystem::create_directories(path);
  if(!boost::filesystem::is_directory(path))
  {
    assert(!"Can't create temporary directory (needed to dump clang headers)");
    abort();
  }


  // TODO: add check for folder creation
  return path.string();
}
