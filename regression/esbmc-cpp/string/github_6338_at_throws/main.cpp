// github #6338: std::string::at must throw a catchable std::out_of_range.
//
// Verified against clang++ -std=c++17 -fsanitize=address,undefined: exits 0.
#include <string>
#include <vector>
#include <stdexcept>
#include <exception>
#include <cassert>

int main()
{
  // Control: vector::at throws and the handler runs.
  std::vector<int> v;
  v.push_back(1);
  bool caught_vector = false;
  try
  {
    v.at(5);
  }
  catch (const std::out_of_range &)
  {
    caught_vector = true;
  }
  assert(caught_vector);

  // string::at behaves the same way.
  std::string s = "ab";
  bool caught_string = false;
  bool fell_through = false;
  try
  {
    s.at(5);
    fell_through = true;
  }
  catch (const std::out_of_range &)
  {
    caught_string = true;
  }
  assert(caught_string);
  assert(!fell_through);

  // The exception is catchable by each of its bases.
  bool caught_logic = false;
  try
  {
    s.at(2);
  }
  catch (const std::logic_error &)
  {
    caught_logic = true;
  }
  assert(caught_logic);

  bool caught_base = false;
  try
  {
    s.at(2);
  }
  catch (const std::exception &)
  {
    caught_base = true;
  }
  assert(caught_base);

  // In-range access does not throw.
  bool caught_in_range = false;
  try
  {
    assert(s.at(0) == 'a');
    assert(s.at(1) == 'b');
  }
  catch (const std::out_of_range &)
  {
    caught_in_range = true;
  }
  assert(!caught_in_range);

  return 0;
}
