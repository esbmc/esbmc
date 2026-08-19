// [ostream.rvalue] as amended by LWG 1203: inserting into an rvalue stream
// returns the *derived* stream, not basic_ostream, so `ostringstream{} << x`
// stays an ostringstream and .str() on the result compiles. ESBMC's own
// src/util/message/message.h builds its timestamp exactly that way, so without
// this no header reaching message.h parses. See #5868.
#include <sstream>
#include <string>
#include <cassert>

int main()
{
  std::string s = (std::ostringstream{} << 42).str();
  assert(s == "42");

  std::string t = (std::ostringstream{} << "ab" << "cd").str();
  assert(t == "abcd");
  return 0;
}
