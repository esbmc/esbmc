// Counterpart of github_7540_seek: the position tellp() reports is never
// negative, so asserting that it is must fail on this assertion rather than on
// the model's own stream-position check (#7540).
#include <cassert>
#include <fstream>

int main()
{
  std::ofstream out;
  out.open("offsets.bin", std::ios::out | std::ios::app);
  out.seekp(0, std::ios::end);
  std::streampos end = out.tellp();
  assert(end < 0);
  return 0;
}
