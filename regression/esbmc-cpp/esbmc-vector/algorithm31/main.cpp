// unique algorithm example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

bool myfunction (int i, int j) {
  return (i==j);
}

template<class FwdIt>
FwdIt unique(FwdIt first, FwdIt last) {
	FwdIt result = first;
	while (++first != last) {
		if (!(*result == *first))
			*(++result) = *first;
	}
	return ++result;
}

template<class FwdIt, class Pr>
FwdIt unique(FwdIt first, FwdIt last, Pr pred) {
	FwdIt result = first;
	while (++first != last) {
		if (!pred(*result, *first))
			*(++result) = *first;
	}
	return ++result;
}

int main () {
  int myints[] = {10,20,20,20,30,30,20,20,10};    // 10 20 20 20 30 30 20 20 10
  vector<int> myvector (myints,myints+9);
  vector<int>::iterator it;

  // using default comparison:
  it = unique (myvector.begin(), myvector.end()); // 10 20 30 20 10 ?  ?  ?  ?
                                                  //                ^
  assert(*it == 30);
  myvector.resize( it - myvector.begin() );       // 10 20 30 20 10

  // using predicate comparison:
  unique (myvector.begin(), myvector.end(), myfunction);   // (no changes)

  // print out content:
  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
