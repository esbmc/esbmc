#include <cassert>
#include <cstring>

namespace std
{

class string {
  public:
    string (char *s)
    {
      _size=strlen(s); //length of string
      str=new char[_size+1]; //increments length of string
      strcpy(str,s);
      str[_size]='\0';
    }

	string ()
	{
		str=new char[1];
		str='\0';
		_size = 0;
	}

	 string(int len)
	 {
	     str = new char [len+1];
	     _size = len;
	 }

	bool operator== ( const char* lhs)
	{
		return (strcmp((char*)lhs, this->str)==0);
	}

    string& operator+ (const string& s)
    {
	   size_t s1 = s._size;
	   assert(s1==1);
#if 0
	    int rhsLen = s._size;
	    int lhsLen = this->_size;
	    int totalLen = lhsLen + rhsLen;
	    assert(totalLen==2);
	    char* temp = new char[totalLen+1];
	    temp = strcpy(temp, s.str);
	    temp = strcat(temp,this->str);
#if 0
	    char temp[totalLen+1];
	    int i, j, k;
	    for (i=0; i<lhsLen; i++)
	        temp[i] = this->str[i];
	    temp[i]='\0';
	    assert(strlen(temp)==1);
	    assert(temp[0]=='X');
	    for(j=i, k=0; j<totalLen; j++, k++)
	    	temp[j] = s.str[k];
	    temp[j]='\0';
	    assert(strlen(temp)==2);
            assert(temp[0]=='X');
            assert(temp[1]=='X');
#endif
	    string next(totalLen);
	    strcpy(next.str, temp);
            assert(strlen(next.str)==2);
            assert(next.str[0]=='X');
	    return next;

#endif
    }
  private:
    size_t _size;
    char* str;
};
}

int main ()
{
  std::string str1("X");
  str1 = str1 + str1;
//  assert(str1=="XX");

  return 0;
}

