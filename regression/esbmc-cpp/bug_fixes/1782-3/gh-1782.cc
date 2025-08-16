#include <iostream>
#include <cassert>
#include <typeinfo>

// Note: We need to use std::cout since 'cout' is redefined below
using std::endl;

template <typename> struct a; // Forward declaration

struct b { 
    a<char> *c;
    
    // Constructor
    b() : c(nullptr) {
        std::cout << "b constructor: pointer initialized to nullptr" << endl;
    }
    
    explicit b(a<char>* ptr) : c(ptr) {
        std::cout << "b constructor: pointer initialized to " << ptr << endl;
    }
    
    // Methods to work with the pointer
    void set_pointer(a<char>* ptr) {
        c = ptr;
        std::cout << "b::set_pointer: pointer set to " << ptr << endl;
    }
    
    a<char>* get_pointer() const {
        return c;
    }
    
    bool has_pointer() const {
        return c != nullptr;
    }
    
    void print_info() const {
        std::cout << "b object: pointer = " << c;
        if (c) {
            std::cout << " (valid)";
        } else {
            std::cout << " (null)";
        }
        std::cout << endl;
    }
};

template <typename T> 
struct a {
    // Add some functionality to make it more interesting
    T data;
    int id;
    static int next_id;
    
    a() : data(T()), id(next_id++) {
        std::cout << "a<" << typeid(T).name() << "> constructor: id = " << id << endl;
    }
    
    explicit a(T value) : data(value), id(next_id++) {
        std::cout << "a<" << typeid(T).name() << "> constructor with value: id = " << id << ", data = " << data << endl;
    }
    
    void set_data(T value) {
        data = value;
        std::cout << "a<" << typeid(T).name() << "> id " << id << ": data set to " << data << endl;
    }
    
    T get_data() const {
        return data;
    }
    
    int get_id() const {
        return id;
    }
    
    void print_info() const {
        std::cout << "a<" << typeid(T).name() << "> id " << id << ": data = " << data << endl;
    }
};

// Static member definition
template <typename T>
int a<T>::next_id = 1;

a<char> cout; // Global variable (shadows std::cout)

int main() {
    std::cout << "=== Template Pointer Structure Demo ===" << endl;
    std::cout << "Note: 'cout' is redefined as a<char>, using std::cout for output" << endl;
    
    // Compile-time assertions
    static_assert(sizeof(b) >= sizeof(void*), "b should contain at least a pointer");
    static_assert(sizeof(a<char>) >= sizeof(char), "a<char> should contain at least a char");
    
    std::cout << "\n1. Testing global 'cout' variable:" << endl;
    // The global 'cout' is of type a<char>
    ::cout.set_data('G'); // Using :: to access global cout, not std::cout
    ::cout.print_info();
    assert(::cout.get_data() == 'G' && "Global cout should store 'G'");
    assert(::cout.get_id() == 1 && "Global cout should have id 1 (first created)");
    
    std::cout << "\n2. Creating b objects:" << endl;
    b obj1;
    b obj2(&::cout); // Point to global cout
    
    // Assert initial states
    assert(!obj1.has_pointer() && "obj1 should not have pointer initially");
    assert(obj2.has_pointer() && "obj2 should have pointer to global cout");
    assert(obj2.get_pointer() == &::cout && "obj2 should point to global cout");
    
    obj1.print_info();
    obj2.print_info();
    
    std::cout << "\n3. Creating more a<char> objects:" << endl;
    a<char> char_obj1('A');
    a<char> char_obj2('B');
    a<char> char_obj3('C');
    
    assert(char_obj1.get_data() == 'A' && "char_obj1 should contain 'A'");
    assert(char_obj2.get_data() == 'B' && "char_obj2 should contain 'B'");
    assert(char_obj3.get_data() == 'C' && "char_obj3 should contain 'C'");
    
    std::cout << "\n4. Linking b objects to a<char> objects:" << endl;
    obj1.set_pointer(&char_obj1);
    
    b obj3(&char_obj2);
    b obj4(&char_obj3);
    
    assert(obj1.get_pointer() == &char_obj1 && "obj1 should point to char_obj1");
    assert(obj3.get_pointer() == &char_obj2 && "obj3 should point to char_obj2");
    assert(obj4.get_pointer() == &char_obj3 && "obj4 should point to char_obj3");
    
    obj1.print_info();
    obj3.print_info();
    obj4.print_info();
    
    std::cout << "\n5. Accessing data through pointers:" << endl;
    if (obj1.has_pointer()) {
        std::cout << "obj1 points to a<char> with data: " << obj1.get_pointer()->get_data() << endl;
        std::cout << "obj1 points to a<char> with id: " << obj1.get_pointer()->get_id() << endl;
        
        // Modify data through pointer
        obj1.get_pointer()->set_data('X');
        assert(char_obj1.get_data() == 'X' && "char_obj1 data should be modified to 'X'");
    }
    
    std::cout << "\n6. Testing different template instantiations:" << endl;
    a<int> int_obj(42);
    a<double> double_obj(3.14);
    a<bool> bool_obj(true);
    
    int_obj.print_info();
    double_obj.print_info();
    bool_obj.print_info();
    
    assert(int_obj.get_data() == 42 && "int_obj should contain 42");
    assert(double_obj.get_data() == 3.14 && "double_obj should contain 3.14");
    assert(bool_obj.get_data() == true && "bool_obj should contain true");
    
    // Verify different template instantiations are different types
    assert(typeid(int_obj) != typeid(double_obj) && "Different template instantiations should be different types");
    assert(typeid(char_obj1) != typeid(int_obj) && "a<char> and a<int> should be different types");
    
    std::cout << "\n7. Array of b objects pointing to different a<char> objects:" << endl;
    const int ARRAY_SIZE = 3;
    b b_array[ARRAY_SIZE];
    a<char> char_array[ARRAY_SIZE] = {a<char>('1'), a<char>('2'), a<char>('3')};
    
    // Link each b to corresponding a<char>
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        b_array[i].set_pointer(&char_array[i]);
        assert(b_array[i].get_pointer() == &char_array[i] && "Array elements should be linked correctly");
    }
    
    // Print array contents
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        std::cout << "b_array[" << i << "] -> ";
        if (b_array[i].has_pointer()) {
            b_array[i].get_pointer()->print_info();
        }
    }
    
    std::cout << "\n8. Pointer reassignment:" << endl;
    b flexible_obj(&char_obj1);
    std::cout << "Initial: ";
    flexible_obj.print_info();
    
    flexible_obj.set_pointer(&char_obj2);
    assert(flexible_obj.get_pointer() == &char_obj2 && "Pointer should be reassigned to char_obj2");
    std::cout << "After reassignment: ";
    flexible_obj.print_info();
    
    flexible_obj.set_pointer(nullptr);
    assert(!flexible_obj.has_pointer() && "Pointer should be null after setting to nullptr");
    std::cout << "After setting to null: ";
    flexible_obj.print_info();
    
    std::cout << "\n9. Memory and type verification:" << endl;
    // Verify that b contains a pointer to a<char>
    assert(sizeof(b) >= sizeof(a<char>*) && "b should contain at least a pointer");
    
    // Test pointer arithmetic safety
    a<char>* ptr1 = &char_obj1;
    a<char>* ptr2 = &char_obj2;
    assert(ptr1 != ptr2 && "Different objects should have different addresses");
    
    std::cout << "\n10. Global cout vs std::cout demonstration:" << endl;
    std::cout << "This is std::cout (standard output)" << endl;
    ::cout.set_data('!');
    std::cout << "Global cout now contains: " << ::cout.get_data() << endl;
    std::cout << "Global cout id: " << ::cout.get_id() << endl;
    
    assert(::cout.get_data() == '!' && "Global cout should contain '!'");
    
    std::cout << "\n=== All assertions passed! Template pointer structure works correctly ===" << endl;
    return 0;
}
