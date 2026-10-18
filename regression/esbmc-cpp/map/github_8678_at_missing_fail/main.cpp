// #8678: std::map::at on an absent key throws std::out_of_range; uncaught,
// it terminates the program.
#include <map>

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  return m.at(2);
}
