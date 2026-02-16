#include <iostream>
#include <cassert>

template <typename> struct a;
struct b;
template <typename> struct a { b *c; };
struct b : a<int> {};

int main() {
    std::cout << "Demonstrating circular template inheritance..." << std::endl;
    
    // Create instances of struct b
    b obj1;
    b obj2;
    
    // Initialize the inherited pointer member c to nullptr
    obj1.c = nullptr;
    obj2.c = nullptr;
    
    // Assert initial null state
    assert(obj1.c == nullptr);
    assert(obj2.c == nullptr);
    
    std::cout << "Created two b objects" << std::endl;
    std::cout << "obj1.c = " << obj1.c << std::endl;
    std::cout << "obj2.c = " << obj2.c << std::endl;
    
    // Set up circular references
    obj1.c = &obj2;  // obj1 points to obj2
    obj2.c = &obj1;  // obj2 points to obj1
    
    // Assert circular references are set correctly
    assert(obj1.c == &obj2);
    assert(obj2.c == &obj1);
    assert(obj1.c != nullptr);
    assert(obj2.c != nullptr);
    
    std::cout << "\nAfter setting up circular references:" << std::endl;
    std::cout << "obj1.c points to obj2 at address: " << obj1.c << std::endl;
    std::cout << "obj2.c points to obj1 at address: " << obj2.c << std::endl;
    std::cout << "Address of obj1: " << &obj1 << std::endl;
    std::cout << "Address of obj2: " << &obj2 << std::endl;
    
    // Demonstrate accessing through the circular reference
    std::cout << "\nDemonstrating circular access:" << std::endl;
    std::cout << "obj1.c->c points back to: " << obj1.c->c << std::endl;
    std::cout << "obj2.c->c points back to: " << obj2.c->c << std::endl;
    
    // Assert the circular nature with detailed checks
    assert(obj1.c->c == &obj1);  // obj1 -> obj2 -> obj1
    assert(obj2.c->c == &obj2);  // obj2 -> obj1 -> obj2
    assert(obj1.c->c->c == &obj2);  // obj1 -> obj2 -> obj1 -> obj2
    assert(obj2.c->c->c == &obj1);  // obj2 -> obj1 -> obj2 -> obj1
    
    // Verify the circular nature
    if (obj1.c->c == &obj1) {
        std::cout << "✓ Circular reference confirmed: obj1 -> obj2 -> obj1" << std::endl;
    }
    
    if (obj2.c->c == &obj2) {
        std::cout << "✓ Circular reference confirmed: obj2 -> obj1 -> obj2" << std::endl;
    }
    
    // Show that b inherits from a<int>
    std::cout << "\nType information:" << std::endl;
    std::cout << "sizeof(a<int>): " << sizeof(a<int>) << " bytes" << std::endl;
    std::cout << "sizeof(b): " << sizeof(b) << " bytes" << std::endl;
    
    // Assert inheritance relationship - b should be same size as a<int> since it only inherits
    assert(sizeof(b) == sizeof(a<int>));
    assert(sizeof(b) == sizeof(void*));  // Should be pointer size
    
    // Demonstrate that b can be used as a<int>
    a<int>* base_ptr = &obj1;
    std::cout << "b object used as a<int>* base pointer" << std::endl;
    std::cout << "base_ptr->c points to: " << base_ptr->c << std::endl;
    
    // Assert polymorphic behavior
    assert(base_ptr->c == &obj2);  // Should point to the same object
    assert(base_ptr->c == obj1.c); // Should be the same as accessing directly
    
    // Self-reference example
    b self_ref_obj;
    self_ref_obj.c = &self_ref_obj;  // Point to itself
    
    // Assert self-reference works
    assert(self_ref_obj.c == &self_ref_obj);
    assert(self_ref_obj.c->c == &self_ref_obj);  // Should still point to itself
    assert(self_ref_obj.c->c->c == &self_ref_obj);  // Infinite self-reference
    
    std::cout << "\nSelf-reference example:" << std::endl;
    std::cout << "self_ref_obj.c points to itself: " << (self_ref_obj.c == &self_ref_obj) << std::endl;
    
    // Additional assertions for robustness
    // Test that different objects have different addresses
    assert(&obj1 != &obj2);
    assert(&obj1 != &self_ref_obj);
    assert(&obj2 != &self_ref_obj);
    
    // Test pointer arithmetic and relationships
    assert(obj1.c != obj2.c);  // They point to different objects
    
    // Test that we can chain access safely
    b* chain_ptr = obj1.c->c->c->c;  // obj1 -> obj2 -> obj1 -> obj2 -> obj1
    assert(chain_ptr == &obj1);
    
    // Test base class casting
    a<int>& base_ref = obj1;
    assert(&base_ref == &obj1);  // Should refer to the same object
    assert(base_ref.c == &obj2);  // Should have same c value
    
    std::cout << "\n✅ All assertions passed! The circular inheritance works correctly." << std::endl;
    
    return 0;
}

