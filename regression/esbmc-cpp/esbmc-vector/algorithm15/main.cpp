// swap algorithm example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

template<class Ty>
void swap_esbmc(Ty& a, Ty& b) {
	Ty c(a);
	a = b;
	b = c;
}

int main () {

  int x=10, y=20;                         // x:10 y:20
  swap_esbmc(x,y);                              // x:20 y:10
  assert(x == 20);
  assert(y == 10);
  vector<int> first (4,x), second (6,y);  // first:4x20 second:6x10
  swap_esbmc(first,second);                     // first:6x10 second:4x20
  assert(first[3] == 10);
  cout << "first contains:";
  for (vector<int>::iterator it=first.begin(); it!=first.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
