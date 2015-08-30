// constructing sets
#include <iostream>
#include <set>
#include <cassert>
using namespace std;

bool fncomp (int lhs, int rhs) {return lhs<rhs;}

struct classcomp {
  bool operator() (const int& lhs, const int& rhs) const
  {return lhs<rhs;}
};

int main ()
{
  int myints[]= {10,20,30,40,50};
  set<int> second (myints,myints+5);        // pointers used as iterators
  set<int> fourth (second.begin(), second.end());  // iterator ctor.
  set<int>::iterator it;
  assert(fourth.size() == 5);
  it = fourth.begin();
  assert(*it == 10);
  it = fourth.end();
  it--;
  assert(*it == 50);

  return 0;
}
