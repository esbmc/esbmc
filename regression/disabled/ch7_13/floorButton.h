// Fig. 7.39: floorButton.h
// FloorButton class definition.
#ifndef FLOORBUTTON_H
#define FLOORBUTTON_H

class Elevator;  // forward declaration

class FloorButton {

public:
    FloorButton( int, Elevator & );  // constructor
   ~FloorButton();                   // destructor

   void pressButton();  // press the button
   void resetButton();  // reset the button

private:
   const int floorNumber;  // button's floor number
   bool pressed;           // button state 
   
   // reference to elevator used to summon 
   // elevator to floor
   Elevator &elevatorRef;

}; // end class FloorButton

#endif // FLOORBUTTON_H

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
