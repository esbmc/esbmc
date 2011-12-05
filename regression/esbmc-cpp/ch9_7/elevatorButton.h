// Fig. 9.35: elevatorButton.h
// ElevatorButton class definition.
#ifndef ELEVATORBUTTON_H
#define ELEVATORBUTTON_H

#include "button.h"  // Button class definition

class ElevatorButton : public Button {

public:
   ElevatorButton( Elevator & );  // constructor
   ~ElevatorButton();             // destructor
   void pressButton();            // press the button

}; // end class ElevatorButton

#endif // ELEVATORBUTTON_H

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
