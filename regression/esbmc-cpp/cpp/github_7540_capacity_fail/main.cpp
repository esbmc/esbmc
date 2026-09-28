// The model's ostringstream holds OSTREAM_CAPACITY characters. On a 32-bit
// target, a field this wide made `_len + n` wrap once streamsize became signed,
// so the capacity check passed and the fill ran off the buffer (#7540).
#include <iomanip>
#include <sstream>

int main()
{
  std::ostringstream os;
  os << "abc" << std::setw(2147483647) << "x";
  return 0;
}
