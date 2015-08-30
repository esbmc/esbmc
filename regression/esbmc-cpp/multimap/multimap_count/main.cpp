// map::count
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  std::multimap<char,int> mymm;

  mymm.insert(std::make_pair('x',50));
  mymm.insert(std::make_pair('y',100));
  mymm.insert(std::make_pair('y',150));
  mymm.insert(std::make_pair('y',200));
  mymm.insert(std::make_pair('z',250));
  mymm.insert(std::make_pair('z',300));

  assert(mymm.count('x') == 1);
  assert(mymm.count('y') == 3);
  assert(mymm.count('z') == 2);

  return 0;
}
