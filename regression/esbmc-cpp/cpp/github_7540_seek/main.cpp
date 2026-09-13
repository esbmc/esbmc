// The end of a file can lie past 2 GiB. With a 32-bit signed size or stream
// position the model wrapped it negative, so tellp() after seeking to the end
// reported an invalid stream position (#7540).
#include <cassert>
#include <fstream>

int main()
{
  std::ofstream out;
  out.open("offsets.bin", std::ios::out | std::ios::app);
  out.seekp(0, std::ios::end);
  std::streampos end = out.tellp();
  assert(end >= 0);
  return 0;
}
