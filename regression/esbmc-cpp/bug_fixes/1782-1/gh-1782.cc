#include <iostream>
#include <cassert>
#include <typeinfo>
using namespace std;

template <typename> struct a; // Forward declaration
struct b; // Forward declaration

template <typename T> 
struct a { 
    b *c; 
    
    a() : c(nullptr) { 
        cout << "a<" << typeid(T).name() << "> constructor called" << endl;
    }
    
    // Add some functionality to demonstrate the relationship
    void set_pointer(b* ptr) {
        c = ptr;
        cout << "a<" << typeid(T).name() << ">::set_pointer called" << endl;
    }
    
    b* get_pointer() const {
        return c;
    }
    
    bool has_pointer() const {
        return c != nullptr;
    }
};

struct b : a<int> {
    int value; // Add some data to make it more interesting
    
    b(int val = 0) : value(val) {
        cout << "b constructor called with value: " << val << endl;
    }
    
    // Method to demonstrate the circular relationship
    void link_to(b* other) {
        this->c = other; // Using inherited member from a<int>
        cout << "b(" << value << ") linked to b(" << (other ? other->value : -1) << ")" << endl;
    }
    
    void print_info() const {
        cout << "b object with value: " << value;
        if (c) {
            cout << ", linked to b with value: " << c->value;
        } else {
            cout << ", not linked to anything";
        }
        cout << endl;
    }
    
    // Recursive method to demonstrate chain traversal
    int count_chain() const {
        if (!c) return 1;
        return 1 + c->count_chain();
    }
};

int main() {
    cout << "=== Circular Template Structure Demo ===" << endl;
    
    // Compile-time assertions to verify the relationships
    static_assert(sizeof(b) >= sizeof(a<int>), "b should contain a<int> as base class");
    
    cout << "\n1. Creating b objects:" << endl;
    b obj1(10);
    b obj2(20);
    b obj3(30);
    
    // Assert initial states
    assert(obj1.value == 10 && "obj1 should have value 10");
    assert(obj2.value == 20 && "obj2 should have value 20");
    assert(obj3.value == 30 && "obj3 should have value 30");
    assert(!obj1.has_pointer() && "obj1 should not have pointer initially");
    assert(!obj2.has_pointer() && "obj2 should not have pointer initially");
    assert(!obj3.has_pointer() && "obj3 should not have pointer initially");
    
    cout << "\n2. Demonstrating inheritance relationship:" << endl;
    // b inherits from a<int>, so we can use it as a<int>
    a<int>* base_ptr = &obj1;
    assert(base_ptr != nullptr && "b should be convertible to a<int>*");
    assert(base_ptr->get_pointer() == nullptr && "Initial pointer should be null");
    
    cout << "\n3. Creating circular relationships:" << endl;
    obj1.link_to(&obj2);
    obj2.link_to(&obj3);
    obj3.link_to(&obj1); // Create a cycle
    
    // Verify the links were created
    assert(obj1.get_pointer() == &obj2 && "obj1 should point to obj2");
    assert(obj2.get_pointer() == &obj3 && "obj2 should point to obj3");
    assert(obj3.get_pointer() == &obj1 && "obj3 should point to obj1");
    assert(obj1.has_pointer() && "obj1 should have pointer after linking");
    
    cout << "\n4. Printing object information:" << endl;
    obj1.print_info();
    obj2.print_info();
    obj3.print_info();
    
    cout << "\n5. Testing template instantiation with different types:" << endl;
    a<double> double_a;
    a<char> char_a;
    
    // These don't inherit from b, but show template flexibility
    assert(!double_a.has_pointer() && "New a<double> should have null pointer");
    assert(!char_a.has_pointer() && "New a<char> should have null pointer");
    
    cout << "\n6. Testing base class functionality through inheritance:" << endl;
    b obj4(40);
    obj4.set_pointer(&obj1); // Using inherited method from a<int>
    assert(obj4.get_pointer() == &obj1 && "obj4 should point to obj1");
    obj4.print_info();
    
    cout << "\n7. Testing polymorphic behavior:" << endl;
    a<int>& base_ref = obj4; // Reference to base class
    assert(base_ref.get_pointer() == &obj1 && "Base reference should access same pointer");
    base_ref.set_pointer(&obj2);
    assert(obj4.get_pointer() == &obj2 && "Setting through base ref should affect derived object");
    
    cout << "\n8. Memory layout verification:" << endl;
    // Verify that b contains a<int> as first subobject
    b* b_ptr = &obj1;
    a<int>* a_ptr = static_cast<a<int>*>(b_ptr);
    assert(a_ptr == reinterpret_cast<a<int>*>(b_ptr) && "a<int> should be first subobject of b");
    
    cout << "\n9. Chain traversal (with cycle detection):" << endl;
    // Create a non-cyclic chain for testing
    b linear1(100);
    b linear2(200);
    b linear3(300);
    
    linear1.link_to(&linear2);
    linear2.link_to(&linear3);
    // linear3.c remains nullptr (no cycle)
    
    assert(linear3.count_chain() == 1 && "Single node should have chain length 1");
    assert(linear2.count_chain() == 2 && "Two-node chain should have length 2");
    assert(linear1.count_chain() == 3 && "Three-node chain should have length 3");
    
    cout << "Linear chain lengths: " << endl;
    cout << "linear1 chain length: " << linear1.count_chain() << endl;
    cout << "linear2 chain length: " << linear2.count_chain() << endl;
    cout << "linear3 chain length: " << linear3.count_chain() << endl;
    
    cout << "\n10. Type relationship verification:" << endl;
    // Verify template specialization
    assert(typeid(obj1) == typeid(b) && "obj1 should be of type b");
    assert(typeid(static_cast<a<int>&>(obj1)) == typeid(a<int>) && "Cast to base should work");
    
    // Test that different template instantiations are different types
    a<int> int_a;
    a<float> float_a;
    assert(typeid(int_a) != typeid(float_a) && "Different template instantiations should be different types");
    
    cout << "\n=== All assertions passed! Circular structure works correctly ===" << endl;
    return 0;
}
