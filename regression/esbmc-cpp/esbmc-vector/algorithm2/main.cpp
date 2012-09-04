// find example
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

template<class InIt, class Ty>
InIt find(InIt first, InIt last, const Ty& value) {
	for (; first != last; first++) {
		if (*first == value) {
			*first = value;
			return first;
		}
	}
	return first;
}

template<class InIt, class Ty>
InIt* find(InIt *first, InIt *last, const Ty& value) {
	for (; first != last; first++) {
		if (*first == value) {
			break;
		}
	}
	return first;
}

int main () {
  int myints[] = { 10, 20, 30 ,40 };
  int * p;

  // pointer to array element:
  p = find(myints,myints+4,30);
  ++p;
  cout << "The element following 30 is " << *p << endl;

  vector<int> myvector (myints,myints+4);
  vector<int>::iterator it;

  // iterator to vector element:
  it = find (myvector.begin(), myvector.end(), 30);
  ++it;
  assert(*it == 40);
  cout << "The element following 30 is " << *it << endl;

  return 0;
}
