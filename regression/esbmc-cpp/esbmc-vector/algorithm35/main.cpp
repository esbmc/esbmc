// rotate algorithm example
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

template<class FwdIt>
void rotate(FwdIt first, FwdIt middle, FwdIt last) {
	FwdIt next = middle;
	while (first != next) {
		swap_esbmc(*first++, *next++);
		if (next == last)
			next = middle;
		else if (first == middle)
			middle = next;
	}
}

int main () {
  vector<int> myvector;
  vector<int>::iterator it;

  // set some values:
  for (int i=1; i<10; ++i) myvector.push_back(i); // 1 2 3 4 5 6 7 8 9

  assert(myvector[0] == 1);
  assert(myvector[6] == 7);

  rotate(myvector.begin(),myvector.begin()+3,myvector.end());
                                                  // 4 5 6 7 8 9 1 2 3
  assert(myvector[0] == 4);
  assert(myvector[6] != 1);
  // print out content:
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
