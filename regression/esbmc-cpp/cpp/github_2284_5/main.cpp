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
    a::vector<d*> f;  // Store pointers instead of objects to avoid recursive instantiation
public:
    void add(d* item) { 
        if (item) f.push_back(item); 
    }
    size_t size() const { return f.size(); }
    const a::vector<d*>& getVector() const { return f; }
    void clear() { f.clear(); }
};

class i : public c {  
    g<i> h;  // Now stores pointers to i, breaking the recursive cycle
public:
    b *e() override {     
        return nullptr;   
    }
    
    void addToCollection(i* item) {
        h.add(item);
    }
    
    size_t getCollectionSize() const {
        return h.size();
    }
    
    void clearCollection() {
        h.clear();
    }
};

class b : public c {  
public:
    b *e() override {     
        return this;      
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
    
    // Test the i class - now using pointers to avoid recursive copying
    i anotherI;
    iObj.addToCollection(&anotherI);  // Pass pointer instead of copy
    a::cout << "Collection size: " << iObj.getCollectionSize() << a::endl;
    
    // Demonstrate polymorphism
    c* cPtr = &bObj;
    b* result = cPtr->e();
    if (result != nullptr) {
        a::cout << "Polymorphic call successful" << a::endl;
    }
    
    // Clean up
    iObj.clearCollection();
    
    a::cout << "Program completed successfully!" << a::endl;
    return 0;
}

