// adjacent_find example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

bool myfunction (int i, int j) {
  return (i==j);
}


template<class FwdIt>
FwdIt adjacent_find(FwdIt first, FwdIt last) {
	if (first != last) {
		FwdIt next = first;
		++next;
		while (next != last) {
			if (*first == *next)
				return first;
			else {
				++first;
				++next;
			}
		}
	}
	return last;
}

template<class FwdIt, class Pr>
FwdIt adjacent_find(FwdIt first, FwdIt last, Pr pred) {
	if (first != last) {
		FwdIt next = first;
		++next;
		while (next != last) {
			if (pred(*first, *next))
				return first;
			else {
				++first;
				++next;
			}
		}
	}
	return last;
}

int main () {
  int myints[] = {10,20,30,30,20,10,10,20};
  vector<int> myvector (myints,myints+8);
  vector<int>::iterator it;

  // using default comparison:
  it = adjacent_find (myvector.begin(), myvector.end());
  assert(*it == 30);
  if (it!=myvector.end())
    cout << "the first consecutive repeated elements are: " << *it << endl;

  //using predicate comparison:
  it = adjacent_find (++it, myvector.end(), myfunction);
  assert(*it != 10);
  if (it!=myvector.end())
    cout << "the second consecutive repeated elements are: " << *it << endl;
  
  return 0;
}
