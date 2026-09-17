// [string.view.modifiers]: remove_prefix(n) is "data_ += n; size_ -= n". The
// model advanced the pointer and left the size alone, so every query after it
// -- size(), end(), find() -- ran off the end of the view.
#include <string_view>
#include <cassert>

int main()
{
  std::string_view sv("hello world");
  sv.remove_prefix(6);
  assert(sv.size() == 5);
  assert(sv[0] == 'w');
  assert(sv.compare(std::string_view("world")) == 0);
  assert(sv.end() - sv.begin() == 5);

  sv.remove_suffix(2);
  assert(sv.size() == 3);
  assert(sv.compare(std::string_view("wor")) == 0);
  return 0;
}
