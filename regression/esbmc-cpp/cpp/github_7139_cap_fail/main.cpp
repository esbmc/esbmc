#include <string>

int main()
{
  // One past capacity(): the guard must reject this rather than write the
  // terminator out of bounds.
  std::string s;
  s.resize(128);

  return 0;
}
