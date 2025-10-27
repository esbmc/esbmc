#include <iostream>
#include <cassert>
#include <typeinfo>
#include <string>
#include <cstdint>
using namespace std;

template <typename a> struct b { 
    a c; 
    b() { /* Default constructor */ }
    b(const a& val) : c(val) { /* Constructor with value */ }
};

struct d; // Forward declaration

struct e { 
    bool called = false; // Track if method was called
    
    void f(const d& obj) { // Use reference to avoid incomplete type error
        cout << "e::f called with d object" << endl;
        called = true;
    }
};

struct g { 
    e h;
    bool initialized = true; // Track initialization
};

struct d : g {
    // d inherits from g, so it has member h of type e
    d() { 
        initialized = true; // Ensure proper initialization
    }
};

struct i { 
    bool method_called = false;
    
    void j(g obj) {
        cout << "i::j called with g object" << endl;
        method_called = true;
        // Can call methods on the g object
        d temp_d;
        obj.h.f(temp_d); // Create a d object and pass to f
        
        // Assert that the nested call worked
        assert(obj.h.called && "e::f should have been called");
    }
};

b<i> k; // Global variable k of type b<i>

int main() {
    cout << "=== C++ Type Relationships Demo with Assertions ===" << endl;
    
    // Compile-time assertions
    static_assert(sizeof(d) >= sizeof(g), "d should be at least as large as g (inheritance)");
    static_assert(sizeof(b<i>) >= sizeof(i), "b<i> should contain an i object");
    
    // Create instances of each type
    d d_obj;
    g g_obj;
    e e_obj;
    i i_obj;
    
    // Assert initial states
    assert(d_obj.initialized && "d object should be initialized");
    assert(g_obj.initialized && "g object should be initialized");
    assert(!e_obj.called && "e::f should not be called yet");
    assert(!i_obj.method_called && "i::j should not be called yet");
    
    cout << "\n1. Using e::f with d object:" << endl;
    e_obj.f(d_obj);
    assert(e_obj.called && "e::f should have been called");
    
    cout << "\n2. Using i::j with g object:" << endl;
    i_obj.j(g_obj);
    assert(i_obj.method_called && "i::j should have been called");
    
    cout << "\n3. Since d inherits from g, we can pass d to i::j:" << endl;
    i i_obj2; // Fresh object to test again
    assert(!i_obj2.method_called && "Fresh i object should not have method_called set");
    
    i_obj2.j(d_obj); // d can be used as g (inheritance)
    assert(i_obj2.method_called && "i::j should work with d object via inheritance");
    
    cout << "\n4. Using the global variable k (type b<i>):" << endl;
    // Test global variable functionality (no null check needed)
    k.c.j(g_obj); // k.c is of type i, so we can call j
    assert(k.c.method_called && "Global k.c should have method_called set");
    
    cout << "\n5. Accessing nested members:" << endl;
    e fresh_e;
    g_obj.h = fresh_e; // Reset to fresh state
    assert(!g_obj.h.called && "Fresh e object should not be called");
    
    g_obj.h.f(d_obj); // g contains e, e can take d
    assert(g_obj.h.called && "Nested call g.h.f(d) should work");
    
    // Test that d inherits g's members correctly
    e another_fresh_e;
    d_obj.h = another_fresh_e; // d inherits h from g
    assert(!d_obj.h.called && "d object should have inherited h member");
    
    d_obj.h.f(d_obj); // d inherits g, so d also has h
    assert(d_obj.h.called && "d.h.f(d) should work via inheritance");
    
    cout << "\n6. Template wrapper demonstration:" << endl;
    b<int> int_wrapper;
    int_wrapper.c = 42;
    assert(int_wrapper.c == 42 && "Template wrapper should store int correctly");
    cout << "int_wrapper.c = " << int_wrapper.c << endl;
    
    b<string> string_wrapper;
    string_wrapper.c = "Hello, World!";
    assert(string_wrapper.c == "Hello, World!" && "Template wrapper should store string correctly");
    cout << "string_wrapper.c = " << string_wrapper.c << endl;
    
    // Test template constructor
    b<int> int_wrapper2(100);
    assert(int_wrapper2.c == 100 && "Template constructor should work");
    
    cout << "\n7. Type relationship assertions:" << endl;
    // Test that d* can be cast to g* (inheritance relationship)
    d* d_ptr = &d_obj;
    g* g_ptr = static_cast<g*>(d_ptr);
    assert(g_ptr != nullptr && "d should be convertible to g via inheritance");
    assert(g_ptr->initialized && "Converted pointer should access g members");
    
    // Test polymorphic behavior
    g& g_ref = d_obj; // d can be referenced as g
    assert(g_ref.initialized && "Reference conversion should work");
    
    cout << "\n8. Memory layout verification:" << endl;
    // Verify that d contains g as a subobject (simplified version)
    g* g_ptr_from_d = static_cast<g*>(&d_obj);
    assert(g_ptr_from_d == reinterpret_cast<g*>(&d_obj) && "d should contain g as first subobject");
    assert(g_ptr_from_d->initialized && "Base class subobject should be accessible");
    
    cout << "\n=== All assertions passed! Program is correct ===" << endl;
    return 0;
}
