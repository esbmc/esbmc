// reverse algorithm example
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

template<class BidIt>
void reverse(BidIt first, BidIt last) {
	while ((first != last) && (first != --last))
		swap_esbmc(*first++, *last);
}

template<class BidIt>
void reverse(BidIt *first, BidIt *last) {
	while ((first != last) && (first != --last))
		swap_esbmc(*first++, *last);
}

int main () {
  vector<int> myvector;
  vector<int>::iterator it;
  int i,k;

  // set some values:
  for (int i=1; i<10; ++i) myvector.push_back(i); // 1 2 3 4 5 6 7 8 9

  reverse(myvector.begin(),myvector.end());       // 9 8 7 6 5 4 3 2 1
  for (i=9, k=0; i>0; i--, k++) assert(myvector[k] != i);
  // print out content:
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
