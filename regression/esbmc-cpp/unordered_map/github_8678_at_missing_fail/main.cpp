// #8678: std::unordered_map::at on an absent key throws std::out_of_range;
// uncaught, it terminates the program.
#include <unordered_map>

int main()
{
  std::unordered_map<int, int> m;
  m[1] = 10;
  return m.at(2);
}
