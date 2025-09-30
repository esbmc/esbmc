#include <module_locator.h>
#include <boost/filesystem.hpp>

namespace bfs = boost::filesystem;

module_locator::module_locator(std::string output_dir)
  : out_dir_(std::move(output_dir))
{
}

std::vector<std::string> module_locator::split(const std::string &s, char delim)
{
  std::vector<std::string> out;
  std::string cur;
  for (char c : s)
  {
    if (c == delim)
    {
      if (!cur.empty())
        out.push_back(cur);
      cur.clear();
    }
    else
    {
      cur.push_back(c);
    }
  }
  if (!cur.empty())
    out.push_back(cur);
  return out;
}

std::string
module_locator::module_path(const std::string &qualified_module) const
{
  const auto parts = split(qualified_module, '.');
  bfs::path p(out_dir_);
  if (parts.empty())
    return (p / (qualified_module + ".json")).string();
  for (size_t i = 0; i + 1 < parts.size(); ++i)
    p /= parts[i];
  p /= parts.back() + std::string(".json");
  return p.string();
}

std::ifstream
module_locator::open_module_file(const std::string &qualified_module) const
{
  return std::ifstream(module_path(qualified_module));
}
