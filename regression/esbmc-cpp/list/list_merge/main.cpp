// list::merge
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  list<double> first, second;
  double mydoubles[] = {1.4, 2.2, 2.9};
  list<double> merged; 
  merged.assign(mydoubles,mydoubles+3);
  
  first.push_back (2.2);
  first.push_back (2.9);

  second.push_back (1.4);

  first.sort();

  first.merge(second);
  assert(first == merged);

  return 0;
}
