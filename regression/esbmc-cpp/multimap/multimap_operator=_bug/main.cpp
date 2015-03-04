// assignment operator with maps
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  std::multimap<char,int> foo,bar;

  foo.insert(std::make_pair('x',32));
  foo.insert(std::make_pair('y',64));
  foo.insert(std::make_pair('y',96));
  foo.insert(std::make_pair('z',128));

  bar = foo;                  // bar now contains 4 ints
  assert(foo != bar);
  foo.clear();                // and first is now empty

  std::cout << "Size of foo: " << foo.size() << '\n';
  std::cout << "Size of bar: " << bar.size() << '\n';
  return 0;
}
