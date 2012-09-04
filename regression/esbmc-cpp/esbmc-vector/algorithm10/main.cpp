// equal algorithm example
#include <iostream>
#include <cassert>
#include <vector>
using namespace std;

bool mypredicate (int i, int j) {
  return (i==j);
}

template<class InIt1, class InIt2>
bool equal(InIt1 first1, InIt1 last1, InIt2 first2) {
	while (first1 != last1) {
		if (!(*first1 == *first2))
			return false;
		++first1;
		++first2;
	}
	return true;
}

template<class InIt1, class InIt2>
bool equal(InIt1 first1, InIt1 last1, InIt2 *first2) {
	while (first1 != last1) {
		if (!(*first1 == *first2))
			return false;
		++first1;
		++first2;
	}
	return true;
}

template<class InIt1, class InIt2, class Pr>
bool equal(InIt1 first1, InIt1 last1, InIt2 first2, Pr pred) {
	while (first1 != last1) {
		if (!pred(*first1, *first2))
			return false;
		++first1;
		++first2;
	}
	return true;
}

int main () {

  int myints[] = {20,40,60,80,100};          //   myints: 20 40 60 80 100
  vector<int>myvector (myints,myints+5);     // myvector: 20 40 60 80 100
  assert((equal (myvector.begin(), myvector.end(), myints)));
  // using default comparison:
  if (equal (myvector.begin(), myvector.end(), myints))
    cout << "The contents of both sequences are equal." << endl;
  else
    cout << "The contents of both sequences differ." << endl;

  myvector[3]=81;                            // myvector: 20 40 60 81 100
  
  return 0;
}
