#include <vector>
#include <iostream>
#include <memory>

namespace a = std;

class b;

class c {
public:
    virtual ~c() = default;  // Virtual destructor for proper cleanup
    virtual b *e() = 0;      // Pure virtual function
};

template <class d> 
class g { 
    a::vector<d> f; 
public:
    void add(const d& item) { f.push_back(item); }
    size_t size() const { return f.size(); }
    const a::vector<d>& getVector() const { return f; }
};

class i : public c {  // Added public inheritance
    g<i> h;
public:
    b *e() override {     // Implementation of virtual function
        return nullptr;   // Simple implementation
    }
    
    void addToCollection(const i& item) {
        h.add(item);
    }
    
    size_t getCollectionSize() const {
        return h.size();
    }
};

class b : public c {  // Added public inheritance
public:
    b *e() override {     // Override virtual function and provide implementation
        return this;      // Return pointer to self
    }
};

int main() {
    a::cout << "Creating objects..." << a::endl;
    
    // Create instances
    b bObj;
    i iObj;
    
    // Test the b class
    b* bPtr = bObj.e();
    if (bPtr != nullptr) {
        a::cout << "b::e() returned a valid pointer" << a::endl;
    }
    
    // Test the i class
    i anotherI;
    iObj.addToCollection(anotherI);
    a::cout << "Collection size: " << iObj.getCollectionSize() << a::endl;
    
    // Demonstrate polymorphism
    c* cPtr = &bObj;
    b* result = cPtr->e();
    if (result != nullptr) {
        a::cout << "Polymorphic call successful" << a::endl;
    }
    
    a::cout << "Program completed successfully!" << a::endl;
    return 0;
}

