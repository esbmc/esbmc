// unique_copy example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool myfunction (int i, int j) {
  return (i==j);
}

template<class InIt, class OutIt>
OutIt unique_copy_esbmc(InIt first, InIt last, OutIt dest) {
	InIt value = first;
	dest = first;
	while (++first != last) {
		if (!(value == *first))
			*(++dest) = value = *first;
	}
	return ++dest;
}

template<class InIt, class OutIt>
OutIt unique_copy_esbmc(InIt *first, InIt *last, OutIt dest) {
	InIt value = *first;
	*dest = *first;
	while (++first != last) {
		if (!(value == *first))
			*(++dest) = value = *first;
	}
	return ++dest;
}

template<class InIt, class OutIt, class Pr>
OutIt unique_copy_esbmc(InIt first, InIt last, OutIt dest, Pr pred) {
#if 0
	InIt value = first;
	dest = first;
	while (++first != last) {
		if (!(pred(value, first)))
			*(++dest) = value = *first;
	}
	return ++dest;
#endif
}

int main () {
  int myints[] = {10,20,20,20,30,30,20,20,10};
  int myints1[] = {10,10,20,20,30,0,0,0,0};
  vector<int> myvector (9);                            // 0  0  0  0  0  0  0  0  0
  vector<int>::iterator it;

  // using default comparison:
  it=unique_copy_esbmc (myints,myints+9,myvector.begin());   // 10 20 30 20 10 0  0  0  0
                                                       //                ^
  assert(*it == 0);
  myvector.assign(myints1, myints1+9);                          // 10 10 20 20 30 0  0  0  0
                                                       //                ^

  // using predicate comparison:
  it=unique_copy_esbmc (myvector.begin(), it, myvector.begin(), myfunction);
                                                       // 10 20 30 20 30 0  0  0  0
                                                       //          ^

  myvector.resize( it - myvector.begin() );            // 10 20 30

  cout << endl;

  return 0;
}
