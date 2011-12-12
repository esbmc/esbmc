// Fig. 15.3: listnode.h
// Template ListNode class definition.
#ifndef LISTNODE_H
#define LISTNODE_H

// forward declaration of class List 
template< class NODETYPE > class List;  

template< class NODETYPE >
class ListNode {
   friend class List< NODETYPE >; // make List a friend

public:
   ListNode( const NODETYPE & );  // constructor
   NODETYPE getData() const;      // return data in node

private:
   NODETYPE data;                 // data
   ListNode< NODETYPE > *nextPtr; // next node in list

}; // end class ListNode

// constructor
template< class NODETYPE >
ListNode< NODETYPE >::ListNode( const NODETYPE &info )
   : data( info ), 
     nextPtr( 0 ) 
{ 
   // empty body

} // end ListNode constructor

// return copy of data in node
template< class NODETYPE >
NODETYPE ListNode< NODETYPE >::getData() const 
{ 
   return data; 
   
} // end function getData

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