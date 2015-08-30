// Fig. 9.16: circletest3.cpp
// Testing class Circle3.
#include <iostream>  

using std::cout;
using std::endl;
using std::fixed;

#include <iomanip>

using std::setprecision;

#include "circle3.h"  // Circle3 class definition

int main()
{
   Circle3 circle( 37, 43, 2.5 ); // instantiate Circle3 object

   // display point coordinates
   cout << "X coordinate is " << circle.getX() 
        << "\nY coordinate is " << circle.getY()
        << "\nRadius is " << circle.getRadius();

   circle.setX( 2 );          // set new x-coordinate
   circle.setY( 2 );          // set new y-coordinate
   circle.setRadius( 4.25 );  // set new radius

   // display new point value
   cout << "\n\nThe new location and radius of circle are\n";
   circle.print();

   // display floating-point values with 2 digits of precision
   cout << fixed << setprecision( 2 );

   // display Circle3's diameter
   cout << "\nDiameter is " << circle.getDiameter();

   // display Circle3's circumference
   cout << "\nCircumference is " << circle.getCircumference();

   // display Circle3's area
   cout << "\nArea is " << circle.getArea();

   cout << endl;

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