// [cctype.syn]/[ctime.syn]/[clocale.syn] put these names in namespace std; the
// models only delegated to the C header, so std::isdigit and friends did not
// resolve even though ::isdigit did (github #5868).
#include <cctype>
#include <ctime>
#include <clocale>
#include <cassert>

int main()
{
  assert(std::isdigit('5'));
  assert(!std::isdigit('a'));
  assert(std::isalpha('a'));
  assert(std::isalnum('5'));
  assert(std::isupper('A'));
  assert(!std::isupper('a'));
  assert(std::islower('a'));
  assert(std::isspace(' '));
  assert(!std::isspace('x'));
  assert(std::isxdigit('f'));
  assert(std::ispunct('!'));
  assert(std::isprint('a'));
  assert(!std::iscntrl('a'));
  assert(std::isgraph('a'));
  assert(std::tolower('A') == 'a');
  assert(std::toupper('a') == 'A');

  std::time_t t = std::time(0);
  (void)t;
  std::clock_t c = std::clock();
  (void)c;
  std::tm bt;
  (void)bt;
  assert(std::difftime(20, 8) == 12.0);

  const char *l = std::setlocale(LC_ALL, "C");
  (void)l;
  std::lconv *lc = std::localeconv();
  (void)lc;
  return 0;
}
