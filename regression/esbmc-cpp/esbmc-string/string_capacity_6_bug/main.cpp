#include <string>
#include <iostream>
#include <cassert>

using namespace std;

int main(){
	string str;
	str = "Test string";
	assert(str.capacity() < 11);
	cout << str.capacity();
	str += ": x";
	assert(str.capacity() < 13);
	cout << str.capacity();
	str.assign("New test string: xyzabcdefg");
	assert(str.capacity() < 26);
	cout << str.capacity();
}
