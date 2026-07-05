#include <exception>

// With no handler installed, std::terminate uses the library default, which is
// a hard failure ("terminate called after throwing an exception").
int main()
{
  std::terminate();
  return 0;
}
