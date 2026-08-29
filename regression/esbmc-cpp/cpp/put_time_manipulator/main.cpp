// [ext.manip]: put_time returns an unspecified type insertable into an
// ostream. ESBMC's own src/util/message/message.h uses it, so without it no
// header reaching message.h parses. See #5868.
#include <iomanip>
#include <iostream>
#include <ctime>
#include <cassert>

int main()
{
  std::time_t t = std::time(0);
  std::tm *lt = std::localtime(&t);
  std::cout << std::put_time(lt, "%Y-%m-%d %H:%M:%S") << std::endl;
  std::cout << std::setw(4) << std::put_time(lt, "%H") << std::endl;
  assert(lt != 0 || lt == 0);
  return 0;
}
