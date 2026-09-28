// #7797: <stop_token>. A source hands out tokens, a callback runs when stop is
// requested, and one registered after the request runs immediately.
#include <stop_token>
#include <cassert>

int ran;

struct marker
{
  int bit;
  explicit marker(int b) : bit(b)
  {
  }
  void operator()() const
  {
    ran |= bit;
  }
};

int main()
{
  std::stop_source src;
  std::stop_token tok = src.get_token();

  assert(src.stop_possible());
  assert(tok.stop_possible());
  assert(!src.stop_requested());
  assert(!tok.stop_requested());

  std::stop_callback<marker> before(tok, marker(1));
  assert(ran == 0);

  assert(src.request_stop());
  assert(ran == 1);
  assert(src.stop_requested());
  assert(tok.stop_requested());
  // A second request changes nothing.
  assert(!src.request_stop());

  // Registered after the request: runs on construction.
  std::stop_callback<marker> after(tok, marker(2));
  assert(ran == 3);

  std::stop_token copy = tok;
  assert(copy.stop_requested());
  assert(copy == tok);

  std::stop_source none(std::nostopstate);
  assert(!none.stop_possible());
  assert(!none.stop_requested());
  assert(!none.request_stop());

  std::stop_token empty;
  assert(!empty.stop_possible());
  assert(!empty.stop_requested());
  assert(!(empty == tok));
  return 0;
}
