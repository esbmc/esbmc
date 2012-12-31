// Fig. 9.37: floorButton.h
// FloorButton class definition.
#ifndef FLOORBUTTON_H
#define FLOORBUTTON_H

#include "button.h"  // Button class definition

class FloorButton : public Button {

public:
    FloorButton( int, Elevator & );  // constructor
   ~FloorButton();                   // destructor
   void pressButton();               // press the button

private:
   const int floorNumber;  // button's floor number

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
