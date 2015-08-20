//TEST FAILS

#include <string>
#include <iostream>
#include <cassert>
using namespace std;
int main(){
	string str;
	str = string("A",12);
	assert(str.capacity() < 12);
	cout << str.capacity() ;	
}

