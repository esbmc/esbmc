// Program-wide handler state: a terminate handler installed in one translation
// unit (helper.cpp) must be seen by std::terminate() called in this one. This
// only holds if __gnu_cxx::__terminate_handler is a single program-wide object
// (external linkage), not a per-TU static.
#include <exception>

void install_handler(); // defined in helper.cpp

int main()
{
  install_handler();
  std::terminate(); // runs the handler installed in the other TU -> exit(0)
  return 1;         // unreachable if the handler is seen program-wide
}
