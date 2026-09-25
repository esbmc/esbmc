#include <string>
#include <iostream>
#include <cassert>
using namespace std;
int main(){
	string str;
	str = string(12,'A');
	assert(str.capacity() >= 12);
	cout << str.capacity() ;	
}

