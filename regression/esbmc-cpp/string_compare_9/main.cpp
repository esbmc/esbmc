#include <string>
#include <cassert>
using namespace std;


int main(){
	string str1, str2, str3;
	int i, j, k;
	
	str1 = string("art");
	str2 = string("earth");
	str3 = string("artist");
	
	i = str2.compare(1, 3, str1);
	j = str3.compare(0, 3, str1);
	k = str2.compare(1, 3, str3, 0, 3);
	
	assert( ( i == 0 ) && ( j == 0 ) && ( k == 0 ) );

}
