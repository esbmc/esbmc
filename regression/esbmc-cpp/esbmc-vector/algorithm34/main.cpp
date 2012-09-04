// reverse_copy example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

template<class BidIt, class OutIt>
OutIt reverse_copy(BidIt first, BidIt last, OutIt dest) {
	while (first != last)
		*dest++ = *--last;
	return dest;
}

template<class BidIt, class OutIt>
OutIt reverse_copy(BidIt *first, BidIt *last, OutIt dest) {
	while (first != last)
		*dest++ = *--last;
	return dest;
}

int main () {
  int myints[] ={1,2,3,4,5,6,7,8,9};
  vector<int> myvector;
  vector<int>::iterator it;
  int i,k;

  myvector.resize(9);

  reverse_copy (myints, myints+9, myvector.begin());
  for(i=0,k=9;i<9;i++,k--) assert(myvector[i] == k);
  // print out content:
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
