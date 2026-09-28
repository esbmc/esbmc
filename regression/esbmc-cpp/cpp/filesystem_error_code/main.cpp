#include <filesystem>
#include <cassert>

int main()
{
  // [filesystem.syn] declares an error_code& overload of nearly every
  // operation, so <filesystem> alone must make std::error_code visible.
  std::error_code ec;
  assert(!ec);
  assert(ec.value() == 0);

  ec = std::make_error_code(std::errc::invalid_argument);
  assert(ec.value() == static_cast<int>(std::errc::invalid_argument));
  return 0;
}
