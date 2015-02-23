// map::size
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  std::multimap<char,int> mymultimap;

  mymultimap.insert(std::make_pair('x',100));
  mymultimap.insert(std::make_pair('y',200));
  mymultimap.insert(std::make_pair('y',350));
  mymultimap.insert(std::make_pair('z',500));

  assert(mymultimap.size() != 4);
  std::cout << "mymultimap.size() is " << mymultimap.size() << '\n';

  return 0;
}
