// Anti-vacuity twin of put_time_manipulator. The manipulator inserts nothing
// observable -- the stream models do not render text -- so there is no output
// to falsify; what this pins is that assertions after the insertion are still
// reachable and can fail, i.e. the manipulator does not make its enclosing
// block vacuous.
#include <iomanip>
#include <iostream>
#include <ctime>
#include <cassert>

int main()
{
  std::time_t t = std::time(0);
  std::tm *lt = std::localtime(&t);
  std::cout << std::put_time(lt, "%H") << std::endl;
  assert(lt == 0);
  return 0;
}
