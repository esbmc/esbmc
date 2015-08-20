//TEST FAILS

#include <string>
#include <cassert>
using namespace std;

int main(){
	string str1, str2;
	int i, j, k;
	str1 = 'T';
	str2 = string();
	i = str1.compare("T");
	j = str1.compare(str2);
	k = str1.compare("Test");
	assert( ( i != 0 ) && ( j <= 0 ) && ( k >= 0) );
}
