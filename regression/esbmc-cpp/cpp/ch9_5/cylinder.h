// Fig. 9.22: cylinder.h
// Cylinder class inherits from class Circle4.
#ifndef CYLINDER_H
#define CYLINDER_H

#include "circle4.h"  // Circle4 class definition

class Cylinder : public Circle4 {

public:

   // default constructor
   Cylinder( int = 0, int = 0, double = 0.0, double = 0.0 ); 
      
   void setHeight( double );  // set Cylinder's height
   double getHeight() const;  // return Cylinder's height 

   double getArea() const;    // return Cylinder's area
   double getVolume() const;  // return Cylinder's volume
   void print() const;        // output Cylinder

private:
   double height;  // Cylinder's height

}; // end class Cylinder

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