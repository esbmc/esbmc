// constructing multimaps
#include <iostream>
#include <map>
#include <cassert>

bool fncomp (char lhs, char rhs) {return lhs<rhs;}

struct classcomp {
  bool operator() (const char& lhs, const char& rhs) const
  {return lhs<rhs;}
};

int main ()
{
  std::multimap<char,int> first;
  int number;

  first.insert(std::pair<char,int>('a',10));
  first.insert(std::pair<char,int>('b',15));
  first.insert(std::pair<char,int>('b',20));
  first.insert(std::pair<char,int>('c',25));

  number = 10;
  for(std::multimap<char,int>::iterator it = first.begin(); it != first.end(); it++, number = number + 5){
    assert( it->second == number);
  }
  assert(first.size() == 4);

  std::multimap<char,int> second (first.begin(),first.end());

  number = 10;
  for(std::multimap<char,int>::iterator it = second.begin(); it != second.end(); it++, number = number + 5){
    assert( it->second == number);
  }
  assert(second.size() == 4);

  std::multimap<char,int> third (second);

  number = 10;
  for(std::multimap<char,int>::iterator it = third.begin(); it != third.end(); it++, number = number + 5){
    assert( it->second == number);
  }
  assert(third.size() == 4);

  std::multimap<char,int,classcomp> fourth;                 // class as Compare

  bool(*fn_pt)(char,char) = fncomp;
  std::multimap<char,int,bool(*)(char,char)> fifth (fn_pt); // function pointer as comp

  return 0;
}
