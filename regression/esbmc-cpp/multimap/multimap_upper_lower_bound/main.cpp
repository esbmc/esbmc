// map::lower_bound/upper_bound
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

int main ()
{
  std::multimap<char,int> mymultimap;
  std::multimap<char,int>::iterator it,itlow,itup;

  mymultimap.insert(std::make_pair('a',10));
  mymultimap.insert(std::make_pair('b',121));
  mymultimap.insert(std::make_pair('c',1001));
  mymultimap.insert(std::make_pair('c',2002));
  mymultimap.insert(std::make_pair('d',11011));
  mymultimap.insert(std::make_pair('e',44));

  itlow = mymultimap.lower_bound ('b');  // itlow points to b
  itup = mymultimap.upper_bound ('d');   // itup points to e (not d)
  assert(itlow->first == 'b');
  assert(itlow->second == 121);
  assert(itup->first == 'e');
  assert(itup->second == 44);

  return 0;
}
