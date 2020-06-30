#include <iostream>
#include <QMap>
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

// must define less than relative to StringClass objects
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
   return strcmp(a.get(), b.get()) < 0;
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
  QMap<StringClass, opposite> QMapObject;

  QMapObject.insert( StringClass("y"), opposite("n") );

  QMap<StringClass, opposite>::iterator it = QMapObject.begin();
  assert(it.key() == StringClass("y"));
  assert(it.value() == opposite("n"));
  assert(QMapObject.size() == 1);

  return 0;
}

