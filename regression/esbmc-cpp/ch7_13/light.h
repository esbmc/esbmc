// Fig. 7.33: light.h
// Light class definition.
#ifndef LIGHT_H
#define LIGHT_H

class Light {

public:
   Light( int );    // constructor 
   ~Light();        // destructor

   void turnOn();   // turns light on
   void turnOff();  // turns light off

private:
   bool on;                // true if on; false if off
   const int floorNumber;  // floor number that contains light

}; // end class Light

#endif // LIGHT_H

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
