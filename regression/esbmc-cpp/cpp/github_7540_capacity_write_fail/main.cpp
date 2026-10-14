// github_7540_capacity_fail for a write rather than a fill: on a 32-bit target
// `_len + n` wraps for a write this large, so the check has to compare n
// with the room left in the model's buffer (#7540).
#include <cstdlib>
#include <sstream>

int main()
{
  const std::streamsize n = 2147483647;
  char *big = static_cast<char *>(malloc(n));
  if (!big)
    return 0;
  std::ostringstream os;
  os << "abc";
  os.write(big, n);
  free(big);
  return 0;
}
