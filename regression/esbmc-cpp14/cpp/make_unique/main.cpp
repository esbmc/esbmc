#include <iostream>
#include <memory>
#include <cassert>

void testCreateUniquePtr()
{
  std::unique_ptr<int> ptr = std::make_unique<int>(10);
  assert(ptr != nullptr); // Check that the pointer is not null
  assert(*ptr == 10);     // Check that the value is correctly assigned
  std::cout << "testCreateUniquePtr passed!" << std::endl;
}

void testTransferOwnership()
{
  std::unique_ptr<int> ptr1 = std::make_unique<int>(20);
  assert(*ptr1 == 20); // Ensure ptr1 owns the resource

  std::unique_ptr<int> ptr2 = std::move(ptr1); // Transfer ownership to ptr2
  assert(ptr1 == nullptr);                     // ptr1 should be null after move
  assert(*ptr2 == 20); // ptr2 should own the resource now

  std::cout << "testTransferOwnership passed!" << std::endl;
}

void testRelease()
{
  std::unique_ptr<int> ptr = std::make_unique<int>(30);
  assert(*ptr == 30); // Check the initial value

  int *rawPtr = ptr.release(); // Release ownership to rawPtr
  assert(ptr == nullptr);      // ptr should be null after release
  assert(*rawPtr == 30);       // rawPtr should point to the original value

  delete rawPtr; // Don't forget to free memory
  std::cout << "testRelease passed!" << std::endl;
}

void testReset()
{
  std::unique_ptr<int> ptr = std::make_unique<int>(40);
  assert(*ptr == 40); // Check the initial value

  ptr.reset(new int(50)); // Reset to point to a new integer
  assert(*ptr == 50);     // Ensure the new value is correct

  ptr.reset();            // Reset to nullptr
  assert(ptr == nullptr); // ptr should be null after reset

  std::cout << "testReset passed!" << std::endl;
}

void testUniquePtrWithCustomDeleter()
{
  struct CustomDeleter
  {
    void operator()(int *ptr) const
    {
      std::cout << "Custom deleter called!" << std::endl;
      delete ptr;
    }
  };

  std::unique_ptr<int, CustomDeleter> ptr(new int(60), CustomDeleter());
  assert(*ptr == 60); // Check that the value is correctly assigned

  std::cout << "testUniquePtrWithCustomDeleter passed!" << std::endl;
}

int main()
{
  testCreateUniquePtr();
  testTransferOwnership();
  testRelease();
  testReset();
  testUniquePtrWithCustomDeleter();

  std::cout << "All tests passed!" << std::endl;
  return 0;
}
