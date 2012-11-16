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
  set<int> third (second);                  // a copy of second
  assert(third.size() == 5);
  it = third.begin();
  assert(*it == 10);
  it = third.end();
  it--;
  assert(*it == 50);
  set<int> fourth (second.begin(), second.end());  // iterator ctor.
  assert(fourth.size() == 5);
  it = fourth.begin();
  assert(*it == 10);
  it = fourth.end();
  it--;
  assert(*it == 50);
  set<int,classcomp> fifth;                 // class as Compare
  assert(fifth.size() == 0);
  bool(*fn_pt)(int,int) = fncomp;
  set<int,bool(*)(int,int)> sixth (fn_pt);  // function pointer as Compare

  return 0;
}
