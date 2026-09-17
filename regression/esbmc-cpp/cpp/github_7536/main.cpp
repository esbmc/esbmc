// libstdc++ and libc++ both reach <exception>, <cstdlib>, <cctype> and <new>
// from <iostream>; the OM tree did not, so programs the host toolchain accepts
// were a PARSING ERROR (#7536).
#include <cassert>
#include <iostream>

void handler()
{
}

struct S
{
  int x;
};

int main()
{
  std::set_terminate(handler);

  char storage[sizeof(S)];
  S *s = new (storage) S();
  s->x = 2;
  assert(s->x == 2);

  assert(isalnum('a'));
  assert(isalpha('a'));
  assert(!isdigit('a'));

  // EXIT_SUCCESS resolves through <cstdlib>; its value is corrected separately.
  int status = EXIT_SUCCESS;
  (void)status;
  return 0;
}
