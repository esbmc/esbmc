#include <cassert>
#include <csetjmp>

std::jmp_buf env;
int steps = 0;

void leave(int code)
{
  ++steps;
  std::longjmp(env, code);
}

int main()
{
  int r = setjmp(env);
  if (r == 0)
    leave(5);
  assert(r == 0);
  assert(steps == 1);
  return 0;
}
