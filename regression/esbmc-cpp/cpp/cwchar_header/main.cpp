#include <cassert>
#include <cwchar>

// [cwchar.syn]. Without the header this is a fatal "'cwchar' file not found",
// which is what blocks most of ESBMC's own translation units (#5868).
//
// The wide-character *functions* have no body in C++ -- a separate,
// pre-existing gap that <wchar.h> shows on its own -- so this pins the names
// and the data handling rather than any return value.

static std::size_t take_size(std::size_t n)
{
  return n;
}

int main()
{
  wchar_t buf[4] = {L'a', L'b', L'c', L'\0'};
  assert(buf[0] == L'a');
  assert(buf[2] == L'c');
  assert(buf[3] == L'\0');

  assert(sizeof(wchar_t) == sizeof(buf[0]));
  assert(take_size(3) == 3);

  std::mbstate_t state;
  (void)state;

  // The declarations resolve in namespace std and at global scope.
  std::size_t (*len)(const wchar_t *) = &std::wcslen;
  assert(len != 0);

  return 0;
}
