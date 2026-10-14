#include <string>

int main()
{
  std::string s("abc");

  // The operational model backs every string with a STRING_CAPACITY buffer, so
  // a resize past it is out of range rather than a reallocation.
  s.resize(200);

  return 0;
}
