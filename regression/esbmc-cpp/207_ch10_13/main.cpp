// Fig. 10.17: circle.cpp
// Circle class member-function definitions.
#include <iostream>  

using std::cout;

#include "circle.h"   // Circle class definition

// default constructor
Circle::Circle( int xValue, int yValue, double radiusValue )
   : Point( xValue, yValue )  // call base-class constructor
{
   setRadius( radiusValue );

} // end Circle constructor

// set radius 
void Circle::setRadius( double radiusValue )
{
   radius = ( radiusValue < 0.0 ? 0.0 : radiusValue );

} // end function setRadius

// return radius 
double Circle::getRadius() const
{
   return radius;

} // end function getRadius

// calculate and return diameter
double Circle::getDiameter() const
{
   return 2 * getRadius();

} // end function getDiameter

// calculate and return circumference
double Circle::getCircumference() const
{
   return 3.14159 * getDiameter();

} // end function getCircumference

// override virtual function getArea: return area of Circle
double Circle::getArea() const
{
   return 3.14159 * getRadius() * getRadius();

} // end function getArea

// override virutual function getName: return name of Circle
string Circle::getName() const
{
   return "Circle";

}  // end function getName

// override virtual function print: output Circle object
void Circle::print() const
{
   cout << "center is ";
   Point::print();  // invoke Point's print function
   cout << "; radius is " << getRadius();

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