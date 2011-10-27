// Fig. 7.21: interface.h
// Header file for interface.cpp

class Implementation;     // forward class declaration

class Interface {

public:
   Interface( int );
   void setValue( int );  // same public interface as
   int getValue() const;  // class Implementation
   ~Interface();

private:

   // requires previous forward declaration (line 4)
   Implementation *ptr;   

}; // end class Interface

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
