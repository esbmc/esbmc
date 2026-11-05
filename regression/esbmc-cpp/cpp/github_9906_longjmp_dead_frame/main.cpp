// #9906: f has returned, so its setjmp is no longer active
#include <csetjmp>
std::jmp_buf b;
void f()
{
  setjmp(b);
}
int main()
{
  f();
  std::longjmp(b, 1);
  return 0;
}
