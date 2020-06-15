#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <new>

template < class T > class Exception
{
public:
  Exception ()
  {
    goto ERROR;
    ERROR:
        ;
  }
};

int
hrandom ()
{
  return random() % 2;
}

class Shape
{
public:
  Shape ()
  {
  }
  ~Shape ()
  {
  }
  void main_reshape ()
  {
  }
  void reshape ()
  {
    int x = hrandom ();
    if (x)
      throw Exception < Shape > ();
  }
};

class Point:public Shape
{
public:
  int x, y;
  Point ()
  {
    x = y = -1;
  }
  Point (int px, int py)
  {
    x = px;
    y = py;
  }
  ~Point ()
  {
  }
  void reshape ()
  {
    int x = hrandom ();
    if (x)
      throw Exception < Point > ();
  }
  void reshapePoint ()
  {
  }
};

// Triangle is derived from Shape. 
class Triangle:public Shape
{
public:
  int a, b, c;
  Triangle (int s1, int s2, int s3)
  {
    a = s1;
    b = s2;
    c = s3;
  }
  void reshape ()
  {
    int x = hrandom ();
    if (x)
      throw Exception < Triangle > ();
  }
};


class Rectangle:public Shape
{
public:
  int l, w;
  Rectangle (int len, int width)
  {
    l = len;
    w = width;
  }
  void reshape ()
  {
    int x = hrandom ();
    if (x)
      throw Exception < Rectangle > ();
  }
};

class Square:public Shape
{
public:
  int e;
  Square (int s)
  {
    e = s;
  }
  virtual void reshape ()
  {
    int x = hrandom ();
    if (x)
      throw Exception < Square > ();
  }
};

int
main ()
{
  try
    {
      Point p (1, 10);
      Triangle t (3, 4, 5);
      Rectangle r (4, 5);
      Square *s = new Square (4);
      
    p.reshape ();
    t.reshape ();
    r.reshape ();
    s->reshape ();
    
    delete s;
    }
  catch (Exception < Point > &ep)
    {
      printf ("Caught a Point\n");
    }
  catch (Exception < Triangle > &et)
    {
      printf ("Caught a Triangle\n");
    }
  catch (Exception < Rectangle > &er)
    {
      printf ("Caught a rectangle\n");
    }
  return 0;
}
