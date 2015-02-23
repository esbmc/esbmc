// Fig. 9.15: circle3.cpp
// Circle3 class member-function definitions.
#include <iostream>  

using std::cout;

#include "circle3.h"   // Circle3 class definition

// default constructor
Circle3::Circle3( int xValue, int yValue, double radiusValue )
{
   x = xValue;
   y = yValue;
   setRadius( radiusValue );

} // end Circle3 constructor

// set radius 
void Circle3::setRadius( double radiusValue )
{
   radius = ( radiusValue < 0.0 ? 0.0 : radiusValue );

} // end function setRadius

// return radius 
double Circle3::getRadius() const
{
   return radius;

} // end function getRadius

// calculate and return diameter
double Circle3::getDiameter() const
{
   return 2 * radius;

} // end function getDiameter

// calculate and return circumference
double Circle3::getCircumference() const
{
   return 3.14159 * getDiameter();

} // end function getCircumference

// calculate and return area
double Circle3::getArea() const
{
   return 3.14159 * radius * radius;

} // end function getArea

// output Circle3 object
void Circle3::print() const
{
   cout << "Center = [" << x << ", " << y << ']'
        << "; Radius = " << radius;

} // end function print

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