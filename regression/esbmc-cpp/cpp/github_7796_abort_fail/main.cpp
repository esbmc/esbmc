// #7796: the program ends abnormally at the raise, with the memory unfreed.
#include <csignal>

int main()
{
  int *p = new int;
  std::signal(SIGABRT, SIG_DFL);
  std::raise(SIGABRT);
  delete p;
  return 0;
}
