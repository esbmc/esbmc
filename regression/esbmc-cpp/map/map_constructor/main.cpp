// constructing maps
#include <iostream>
#include <map>
#include <cassert>
using namespace std;

bool fncomp (char lhs, char rhs) {return lhs<rhs;}

struct classcomp {
  bool operator() (const char& lhs, const char& rhs) const
  {return lhs<rhs;}
};

int main ()
{
  map<char,int> first;

  first['a']=10;
  first['b']=30;
  first['c']=50;
  first['d']=70;

  map<char,int> second (first.begin(),first.end());
  assert(second['a']==10);
  assert(second['b']==30);
  assert(second['c']==50);
  assert(second['d']==70);

  map<char,int> third (second);
  
  assert(third['a']==10);
  assert(third['b']==30);
  assert(third['c']==50);
  assert(third['d']==70);

  map<char,int,classcomp> fourth;                 // class as Compare
  assert(fourth['a']==0);
  assert(fourth['b']==0);
  assert(fourth['c']==0);
  assert(fourth['d']==0);

  bool(*fn_pt)(char,char) = fncomp;
  map<char,int,bool(*)(char,char)> fifth (fn_pt); // function pointer as Compare

  return 0;
}
