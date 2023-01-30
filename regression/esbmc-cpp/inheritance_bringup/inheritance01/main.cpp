/*
 * Multiple inheritance: methods call
 */
#include <cassert>

// Base class Shape
class Shape
{
   public:
      void setWidth(int w){
         width = w;
      }
      void setHeight(int h){
         height = h;
      }
   protected:
      int width;
      int height;
};

// Base class PaintCost
class PaintCost
{
   public:
      int getCost(int area){
         return area * 70;
      }
};

// Derived class
class Rectangle: public Shape, public PaintCost
{
   public:
      int getArea(){
         return (width * height);
      }
};

int main(void)
{
   Rectangle Rect;
   int area;
   int cost;

   Rect.setWidth(5);
   Rect.setHeight(7);

   area = Rect.getArea();
   cost = Rect.getCost(area);
   assert(cost == 2450); // PASS

   return 0;
}
