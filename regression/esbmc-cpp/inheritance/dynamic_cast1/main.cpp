//dynamic cast test
//#include <iostream>
#include <cassert>
//using namespace std;

class CPolygon {
  protected:
    int width, height;
  public:
  CPolygon(){}
    void set_values (int a, int b)
      { width=a; height=b; }
    virtual int area (void) =0;
  };

class CRectangle: public CPolygon {
  public:
  CRectangle(int w, int h)
  {
  	width = w;
  	height = h;
  
  }
    int area (void)
      { return (width * height); }
  };

class CTriangle: public CPolygon {
  public:
  CTriangle(int w, int h)
  {
  	width = w;
  	height = h;
  
  }
    int area (void)
      { return ((width * height) / 2); }
  };
#if 0
class CCircle: public CPolygon {
  public:
  CCircle(int w, int h)
  {
  	width = w;
  	height = h;
  
  }
    int area (void)
      { return ((width / 2)*(width / 2) * 3.1415); }
  };

class CSquare: public CPolygon {
  public:
  CSquare(int w, int h)
  {
  	width = w;
  	height = h;
  
  }
    int area (void)
      { return (width * width); }
  };

#endif

int main () {
	CPolygon* polygons[4];
	
	polygons[0] = new CTriangle(20,25); //CRectangle(20,30);
	polygons[1] = new CRectangle(20,30); //new CTriangle(20,25);
//	polygons[2] = new CCircle(25,25);
//	polygons[3] = new CSquare(18,18);
	
	for(int i = 0; i < 2; i++)
	{
		CTriangle* trin = dynamic_cast <CTriangle *> (polygons[i]);
		if (trin != 0)
		{
			trin->set_values(10, 10);
			assert(trin->area() == 50);
		}

#if 0		
		CCircle* circ = dynamic_cast <CCircle *> (polygons[i]);
		if (circ != 0)
		{
			circ->set_values(10, 10);
			assert(circ->area() == 78);
		}
		
		CSquare* sqrr = dynamic_cast <CSquare *> (polygons[i]);
		if (sqrr != 0)
		{
			sqrr->set_values(10, 10);
			assert(sqrr->area() == 100);
		}
		
		CRectangle* rect = dynamic_cast <CRectangle *> (polygons[i]);
		if (rect != 0)
		{
			rect->set_values(10, 20);
			assert(rect->area() == 200);
		}
#endif		
	}
/*
*/

  return 0;
}
