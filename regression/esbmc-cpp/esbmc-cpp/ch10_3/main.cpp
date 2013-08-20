// Fig. 10.7: fig10_07.cpp
// Attempting to invoke derived-class-only member functions
// through a base-class pointer.
#include "point.h"   // Point class definition
#include "circle.h"  // Circle class definition

int main()
{
   Point *pointPtr = 0;
   Circle circle( 120, 89, 2.7 );

   // aim base-class pointer at derived-class object
   pointPtr = &circle;

   // invoke base-class member functions on derived-class
   // object through base-class pointer
   int x = pointPtr->getX();
   int y = pointPtr->getY();
   pointPtr->setX( 10 );
   pointPtr->setY( 10 );
   pointPtr->print();

   // attempt to invoke derived-class-only member functions
   // on derived-class object through base-class pointer   
   double radius = pointPtr->getRadius();
   pointPtr->setRadius( 33.33 );
   double diameter = pointPtr->getDiameter();
   double circumference = pointPtr->getCircumference();
   double area = pointPtr->getArea();

   return 0;

} // end main

/**************************************************************************
 * (C) Copyright 1992-2003 by Deitel & Associates, Inc. and Prentice      *
 * Hall. All Rights Reserved.                                             *
 *                                                                        *
 * DISCLAIMER: The authors and publisher of this book have used their     *
 * best efforts in preparing the book. These efforts include the          *
 * development, research, and testing of the theories and programs        *
 * to determine their effectiveness. The authors and publisher make       *
 * no warranty of any kind, expressed or implied, with regard to these    *
 * programs or to the documentation contained in these books. The authors *
 * and publisher shall not be liable in any event for incidental or       *
 * consequential damages in connection with, or arising out of, the       *
 * furnishing, performance, or use of these programs.                     *
 *************************************************************************/