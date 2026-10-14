#include <sstream>
#include <iomanip>
#include <cstring>
#include <cassert>

int main()
{
  std::ostringstream ss;
  ss << std::setw(4) << 7;
  assert(strcmp(ss.str().c_str(), "   7") == 0);

  std::ostringstream fs;
  fs << std::setfill('0') << std::setw(3) << 42;
  assert(strcmp(fs.str().c_str(), "042") == 0);

  std::ostringstream ls;
  ls << std::left << std::setw(3) << 5;
  assert(strcmp(ls.str().c_str(), "5  ") == 0);
  return 0;
}
