// Fig. 9.6: pointtest.cpp
// Testing class Point.
#include <iostream>  
#include <cassert>

using std::cout;
using std::endl;

#include "point.h"  // Point class definition

int main()
{
   Point point( 72, 115 );     // instantiate Point object

   // display point coordinates
   cout << "X coordinate is " << point.getX() 
        << "\nY coordinate is " << point.getY();

   point.setX( 10 ); // set x-coordinate
   point.setY( 10 ); // set y-coordinate

   // display new point value
   cout << "\n\nThe new location of point is ";
   point.print();
   cout << endl;

   assert(point.getX() == 10);
   assert(point.getY() == 10);

   return 0;  // indicates successful termination
   
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
