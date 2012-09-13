// generate_n example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

template<class OutIt, class Diff, class Fn0>
void generate_n(OutIt first, Diff n, Fn0 func) {
	for (; n > 0; --n)
		*first++ = func();
}

template<class OutIt, class Diff, class Fn0>
void generate_n(OutIt *first, Diff n, Fn0 func) {
	for (; n > 0; --n)
		*first++ = func();
}

int current(0);
int UniqueNumber () { return ++current; }

int main () {
  int myarray[9];
  int i;

  generate_n (myarray, 9, UniqueNumber);

  cout << "myarray contains:";
  for (int i=0; i<9; ++i)
    cout << " " << myarray[i];

  for (i=0;i<8;i++) assert(myarray[i] == i+1);

  cout << endl;
  return 0;
}
