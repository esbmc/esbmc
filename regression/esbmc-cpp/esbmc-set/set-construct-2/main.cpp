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
  set<int> first;                           // empty set of ints
  assert(first.size() == 0);
  assert(first.begin() == first.end());
  int myints[]= {10,20,30,40,50};
  set<int> second (myints,myints+5);        // pointers used as iterators
  assert(second.size() == 5);
  set<int>::iterator it = second.begin();
  assert(*it == 10);
  it = second.end();
  it--;
  assert(*it == 50);
  return 0;
}
