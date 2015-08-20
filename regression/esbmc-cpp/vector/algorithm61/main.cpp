// min_element/max_element
#include <iostream>
#include <cassert>
using namespace std;

template<class FwdIt>
FwdIt max_element(FwdIt first, FwdIt last) {
	FwdIt largest = first;
	if (first == last)
		return last;
	while (++first != last)
		if (*largest < *first)
			largest = first;
	return largest;
}

template<class FwdIt, class Pr>
FwdIt max_element(FwdIt first, FwdIt last, Pr pred) {
	FwdIt largest = first;
	if (first == last)
		return last;
	while (++first != last)
		if (pred(*largest, *first))
			largest = first;
	return largest;
}

template<class FwdIt, class Pr>
FwdIt* max_element(FwdIt *first, FwdIt *last, Pr pred) {
	FwdIt *largest = first;
	if (first == last)
		return last;
	while (++first != last)
		if (pred(*largest, *first))
			largest = first;
	return largest;
}

int* max_element(int *first, int *last) {
	int *largest = first;
	if (first == last)
		return last;
	while (++first != last)
		if (*largest < *first)
			largest = first;
	return largest;
}

template<class FwdIt>
FwdIt min_element(FwdIt first, FwdIt last) {
	FwdIt lowest = first;
	if (first == last)
		return last;
	while (++first != last)
		if (*first < *lowest)
			lowest = first;
	return lowest;
}

int* min_element(int *first, int *last) {
	int *lowest = first;
	if (first == last)
		return last;
	while (++first != last)
		if (*first < *lowest)
			lowest = first;
	return lowest;
}

template<class FwdIt, class Pr>
FwdIt min_element(FwdIt first, FwdIt last, Pr pred) {
	FwdIt *lowest = first;
	if (first == last)
		return last;
	while (++first != last)
		if (pred(*first, *lowest))
			lowest = first;
	return lowest;
}

template<class FwdIt, class Pr>
FwdIt* min_element(FwdIt *first, FwdIt *last, Pr pred) {
	FwdIt *lowest = first;
	if (first == last)
		return last;
	while (++first != last)
		if (pred(*first, *lowest))
			lowest = first;
	return lowest;
}

bool myfn(int i, int j) { return i<j; }

struct myclass {
  bool operator() (int i,int j) { return i<j; }
} myobj;

int main () {
  int myints[] = {3,7,2,5,6,4,9};

  // using default comparison:
  cout << "The smallest element is " << *min_element(myints,myints+7) << endl;
  assert(*min_element(myints,myints+7) == 2);
  cout << "The largest element is " << *max_element(myints,myints+7) << endl;
  assert(*max_element(myints,myints+7) == 9);

  // using function myfn as comp:
  cout << "The smallest element is " << *min_element(myints,myints+7,myfn) << endl;
  assert(*min_element(myints,myints+7,myfn) == 2);
  cout << "The largest element is " << *max_element(myints,myints+7,myfn) << endl;
  assert(*max_element(myints,myints+7,myfn) == 9);

  // using object myobj as comp:
  cout << "The smallest element is " << *min_element(myints,myints+7,myobj) << endl;
  assert(*min_element(myints,myints+7,myobj) == 2);
  cout << "The largest element is " << *max_element(myints,myints+7,myobj) << endl;
  assert(*max_element(myints,myints+7,myobj) == 9);

  return 0;
}
