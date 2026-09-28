// Negative counterpart of github_3387_newer_headers_c++03: not vacuous.
#include <chrono>
#include <unordered_map>
#include <optional>
#include <string_view>
#include <source_location>

#include <vector>
#include <cassert>

int main()
{
  std::vector<int> v;
  v.push_back(4);
  v.push_back(5);
  assert(v[0] + v[1] == 10);
  return 0;
}
