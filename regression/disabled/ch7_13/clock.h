// Fig. 7.27: clock.h
// Clock class definition.
#ifndef CLOCK_H
#define CLOCK_H

class Clock {

public:
   Clock();              // constructor
   ~Clock();             // destructor
   void tick();          // increment clock by one second
   int getTime() const;  // returns clock's current time

private:
   int time;            // clock's time

}; // end class Clock

#endif // CLOCK_H

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
