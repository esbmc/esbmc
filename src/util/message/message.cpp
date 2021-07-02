/*******************************************************************\

Module: 

Author: Rafael Menezes, rafael.sa.menezes@outlook.com

Maintainers:
\*******************************************************************/

#include <boost/filesystem.hpp>
#include <util/message/message.h>

FILE *messaget::get_temp_file()
{
  // Get the temp file dir
  const boost::filesystem::path tmp_path =
    boost::filesystem::temp_directory_path();

  // Define the pattern for the name
  const std::string pattern = (tmp_path / "esbmc.%%%%-%%%%").string();
  boost::filesystem::path path;

  // Try to get a name that is not used already e.g. esbmc.0000-0000
  do
  {
    path = boost::filesystem::unique_path(pattern);
  } while(
    boost::filesystem::exists(path)); // TODO: This may cause infinite recursion

  // If path folders doesn't exist, create then
  boost::filesystem::create_directories(path);
  // Open File
  FILE *f = fopen(path.string().c_str(), "w+");
  return f;
}

void messaget::insert_and_close_file_contents(VerbosityLevel l, FILE *f) const
{
  const int MAX_LINE_LENGTH = 1024;
  // Go to the beginning of the file
  fseek(f, 0, SEEK_SET);
  char line[MAX_LINE_LENGTH] = {0};

  // Read every line of file
  while(fgets(line, MAX_LINE_LENGTH, f))
    // Send to message print method
    print(l, line);

  // Close the file
  fclose(f);
}
