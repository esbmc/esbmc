#include <iostream>
#include <map>
#include <cstring>
#include <cassert>
using namespace std;

class StringClass {
  char str[20];
public:
  StringClass() { 
     strcpy(str, ""); 
  }
  StringClass(char *s) { 
     strcpy(str, s); 
  }
  char *get() { 
     return str; 
  }
};

bool operator<(StringClass a, StringClass b)
{
   return strcmp(a.get(), b.get()) < 0;
}
bool operator==(StringClass a, StringClass b)
{
   return (strcmp(a.get(), b.get()) == 0);
}
bool operator!=(StringClass a, StringClass b)
{
   return (strcmp(a.get(), b.get()) != 0);
}

class opposite {
  char str[20];
public:
  opposite() { 
     strcmp(str, ""); 
  }
  opposite(char *s) { 
     strcpy(str, s); 
  }
  char *get() { 
     return str; 
  }
};

bool operator<(opposite a, opposite b)
{
   return (strcmp(a.get(), b.get()) < 0);
}
bool operator==(opposite a, opposite b)
{
   return (strcmp(a.get(), b.get()) == 0);
}
bool operator!=(opposite a, opposite b)
{
   return (strcmp(a.get(), b.get()) != 0);
}

int main()
{
  multimap<StringClass, opposite> mapObject;

  mapObject.insert(multimap<StringClass, opposite>::value_type(StringClass("y"),opposite("n")));
  mapObject.insert(multimap<StringClass, opposite>::value_type(StringClass("a"),opposite("b")));
  mapObject.insert(multimap<StringClass, opposite>::value_type(StringClass("c"),opposite("d")));
  multimap<StringClass, opposite>::iterator it = mapObject.begin();
  assert(it->first == StringClass("a"));
  assert(it->second == opposite("d"));
  assert(mapObject.size() == 2);
  return 0;
}
