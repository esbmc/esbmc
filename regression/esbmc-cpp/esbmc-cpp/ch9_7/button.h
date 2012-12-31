// Fig. 9.33: button.h
// Definition for class Button.
#ifndef BUTTON_H
#define BUTTON_H

class Elevator;                 // forward declaration

class Button {

public:
   Button( Elevator & );        // constructor
   ~Button();                   // destructor
   void pressButton();          // sets button on
   void resetButton();          // resets button off

protected:
   
	// reference to button's elevator
   Elevator &elevatorRef;

private:
   bool pressed;                // state of button

}; // end class Button

#endif // BUTTON_H

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
