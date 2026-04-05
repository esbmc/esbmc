// Regression test for https://github.com/esbmc/esbmc/discussions/3978
// ~list should not cause infinite unwinding when main() lacks return 0
#include <list>

int main()
{
  std::list<int> lst;
  lst.push_front(2);
}
