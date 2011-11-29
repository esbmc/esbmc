// Fig. 17.18: tree.h
// Template Tree class definition.
#ifndef TREE_H
#define TREE_H

#include <iostream>

using std::endl;

#include <new>
#include "treenode.h"

template< class NODETYPE >
class Tree {

public:
   Tree();
   void insertNode( const NODETYPE & );
   void preOrderTraversal() const;
   void inOrderTraversal() const;
   void postOrderTraversal() const;

private:
   TreeNode< NODETYPE > *rootPtr;

   // utility functions
   void insertNodeHelper( 
      TreeNode< NODETYPE > **, const NODETYPE & );
   void preOrderHelper( TreeNode< NODETYPE > * ) const;
   void inOrderHelper( TreeNode< NODETYPE > * ) const;
   void postOrderHelper( TreeNode< NODETYPE > * ) const;

}; // end class Tree

// constructor
template< class NODETYPE >
Tree< NODETYPE >::Tree() 
{ 
   rootPtr = 0; 

} // end Tree constructor

// insert node in Tree
template< class NODETYPE >
void Tree< NODETYPE >::insertNode( const NODETYPE &value )
{ 
   insertNodeHelper( &rootPtr, value ); 

} // end function insertNode

// utility function called by insertNode; receives a pointer
// to a pointer so that the function can modify pointer's value
template< class NODETYPE >
void Tree< NODETYPE >::insertNodeHelper( 
   TreeNode< NODETYPE > **ptr, const NODETYPE &value )
{
   // subtree is empty; create new TreeNode containing value
   if ( *ptr == 0 )  
      *ptr = new TreeNode< NODETYPE >( value );

   else  // subtree is not empty

      // data to insert is less than data in current node
      if ( value < ( *ptr )->data )
         insertNodeHelper( &( ( *ptr )->leftPtr ), value );

      else

         // data to insert is greater than data in current node
         if ( value > ( *ptr )->data )
            insertNodeHelper( &( ( *ptr )->rightPtr ), value );

         else  // duplicate data value ignored
            cout << value << " dup" << endl;

} // end function insertNodeHelper

// begin preorder traversal of Tree
template< class NODETYPE > 
void Tree< NODETYPE >::preOrderTraversal() const
{ 
   preOrderHelper( rootPtr ); 

} // end function preOrderTraversal

// utility function to perform preorder traversal of Tree
template< class NODETYPE >
void Tree< NODETYPE >::preOrderHelper( 
   TreeNode< NODETYPE > *ptr ) const
{
   if ( ptr != 0 ) {
      cout << ptr->data << ' ';         // process node
      preOrderHelper( ptr->leftPtr );   // go to left subtree
      preOrderHelper( ptr->rightPtr );  // go to right subtree

   } // end if

} // end function preOrderHelper

// begin inorder traversal of Tree
template< class NODETYPE >
void Tree< NODETYPE >::inOrderTraversal() const
{ 
   inOrderHelper( rootPtr ); 

} // end function inOrderTraversal

// utility function to perform inorder traversal of Tree
template< class NODETYPE >
void Tree< NODETYPE >::inOrderHelper( 
   TreeNode< NODETYPE > *ptr ) const
{
   if ( ptr != 0 ) {
      inOrderHelper( ptr->leftPtr );   // go to left subtree
      cout << ptr->data << ' ';        // process node
      inOrderHelper( ptr->rightPtr );  // go to right subtree

   } // end if

} // end function inOrderHelper

// begin postorder traversal of Tree
template< class NODETYPE >
void Tree< NODETYPE >::postOrderTraversal() const
{ 
   postOrderHelper( rootPtr ); 

} // end function postOrderTraversal

// utility function to perform postorder traversal of Tree
template< class NODETYPE >
void Tree< NODETYPE >::postOrderHelper( 
   TreeNode< NODETYPE > *ptr ) const
{
   if ( ptr != 0 ) {
      postOrderHelper( ptr->leftPtr );   // go to left subtree
      postOrderHelper( ptr->rightPtr );  // go to right subtree
      cout << ptr->data << ' ';          // process node

   } // end if

} // end function postOrderHelper

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