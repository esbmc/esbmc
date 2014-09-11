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
  multiset<int> first;                           // empty set of ints

  int myints[]= {10,20,30,40,50};
  multiset<int> second (myints,myints+5);        // pointers used as iterators

  multiset<int> third (second);                  // a copy of second

  first.~multiset();
  second.~multiset();
  third.~multiset();

  assert(first.size() == 0);
  assert(first.size() == 0);
  assert(first.size() == 0);

  return 0;
}
