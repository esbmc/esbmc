// Every header below postdates C++03 and must be inert, not a parse error,
// under --std c++03, while the C++03 headers sharing the TU keep working
// (#3387).
#include <initializer_list>
#include <chrono>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <any>
#include <optional>
#include <variant>
#include <string_view>
#include <filesystem>
#include <source_location>

#include <vector>
#include <string>
#include <map>
#include <cassert>

int main()
{
  std::vector<int> v;
  v.push_back(4);
  v.push_back(5);
  assert(v[0] + v[1] == 9);

  std::string s = "ab";
  s += "c";
  assert(s.size() == 3);
  assert(s[2] == 'c');

  std::map<int, int> m;
  m[1] = 7;
  assert(m[1] == 7);
  return 0;
}
