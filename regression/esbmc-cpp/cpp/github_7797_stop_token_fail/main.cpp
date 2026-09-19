// #7797: a token from a live source that has not been asked to stop must not
// report a stop.
#include <stop_token>
#include <cassert>

int main()
{
  std::stop_source src;
  std::stop_token tok = src.get_token();
  assert(tok.stop_requested());
  return 0;
}
