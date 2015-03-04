// Fig. 9.19: circle4.h
// Circle4 class contains x-y coordinate pair and radius.
#ifndef CIRCLE4_H
#define CIRCLE4_H

#include "point3.h"  // Point3 class definition

class Circle4 : public Point3 {

public:

   // default constructor
   Circle4( int = 0, int = 0, double = 0.0 );  

   void setRadius( double );   // set radius
   double getRadius() const;   // return radius

   double getDiameter() const;       // return diameter
   double getCircumference() const;  // return circumference
   double getArea() const;           // return area

   void print() const;         // output Circle4 object

private: 
   double radius;  // Circle4's radius

}; // end class Circle4

#endif 

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
