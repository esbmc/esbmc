/*
 * C++ Verification Example
 *
 * Demonstrates ESBMC's C++ verification capabilities:
 * - Class invariant checking
 * - STL container verification
 * - RAII and resource management
 * - Template function verification
 * - Smart pointer safety
 *
 * Run with: esbmc cpp-verify.cpp --unwind 10 --memory-leak-check
 */

#include <cassert>
#include <vector>

int __ESBMC_nondet_int(void);
void __ESBMC_assume(bool);
void __ESBMC_assert(bool, const char *);

/* ============================================
 * Example 1: Class Invariant Verification
 * ============================================ */

class BoundedStack {
    int *data;
    int capacity;
    int top;

public:
    BoundedStack(int cap) : capacity(cap), top(0) {
        __ESBMC_assert(cap > 0, "Capacity must be positive");
        data = new int[cap];
    }

    ~BoundedStack() {
        delete[] data;
    }

    bool is_empty() const { return top == 0; }
    bool is_full() const { return top >= capacity; }
    int size() const { return top; }

    bool push(int val) {
        if (is_full()) return false;
        data[top++] = val;
        __ESBMC_assert(top >= 0 && top <= capacity, "Stack invariant");
        return true;
    }

    int pop() {
        __ESBMC_assert(!is_empty(), "Cannot pop from empty stack");
        int val = data[--top];
        __ESBMC_assert(top >= 0 && top <= capacity, "Stack invariant");
        return val;
    }
};

void class_invariant_example() {
    int cap = __ESBMC_nondet_int();
    __ESBMC_assume(cap > 0 && cap <= 5);

    BoundedStack stack(cap);
    __ESBMC_assert(stack.is_empty(), "New stack is empty");

    int n = __ESBMC_nondet_int();
    __ESBMC_assume(n >= 0 && n <= cap);

    for (int i = 0; i < n; i++) {
        int val = __ESBMC_nondet_int();
        stack.push(val);
    }

    __ESBMC_assert(stack.size() == n, "Size matches push count");

    if (!stack.is_empty()) {
        int old_size = stack.size();
        stack.pop();
        __ESBMC_assert(stack.size() == old_size - 1, "Pop decreases size");
    }
}

/* ============================================
 * Example 2: STL Vector Verification
 * ============================================ */

void vector_verification_example() {
    std::vector<int> v;

    int n = __ESBMC_nondet_int();
    __ESBMC_assume(n > 0 && n <= 5);

    for (int i = 0; i < n; i++) {
        v.push_back(i * 10);
    }

    assert(v.size() == (size_t)n);
    assert(v[0] == 0);

    if (n >= 2) {
        assert(v[1] == 10);
    }

    v.clear();
    assert(v.empty());
}

/* ============================================
 * Example 3: Resource Management (RAII)
 * ============================================ */

class Buffer {
    int *data;
    int size;

public:
    Buffer(int sz) : size(sz) {
        __ESBMC_assert(sz > 0, "Size must be positive");
        data = new int[sz];
        for (int i = 0; i < sz; i++) data[i] = 0;
    }

    ~Buffer() {
        delete[] data;
    }

    // Prevent copying (rule of three)
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    int get(int idx) const {
        __ESBMC_assert(idx >= 0 && idx < size, "Index in bounds");
        return data[idx];
    }

    void set(int idx, int val) {
        __ESBMC_assert(idx >= 0 && idx < size, "Index in bounds");
        data[idx] = val;
    }

    int length() const { return size; }
};

void raii_example() {
    int sz = __ESBMC_nondet_int();
    __ESBMC_assume(sz > 0 && sz <= 10);

    Buffer buf(sz);

    int idx = __ESBMC_nondet_int();
    __ESBMC_assume(idx >= 0 && idx < sz);

    buf.set(idx, 42);
    __ESBMC_assert(buf.get(idx) == 42, "Value stored correctly");
    // Buffer automatically freed by destructor -- no leak
}

/* ============================================
 * Example 4: Template Function Verification
 * ============================================ */

template<typename T>
T safe_max(T a, T b) {
    return (a > b) ? a : b;
}

template<typename T>
T safe_min(T a, T b) {
    return (a < b) ? a : b;
}

template<typename T>
T clamp(T val, T lo, T hi) {
    __ESBMC_assert(lo <= hi, "lo must not exceed hi");
    return safe_max(lo, safe_min(val, hi));
}

void template_example() {
    int a = __ESBMC_nondet_int();
    int b = __ESBMC_nondet_int();
    __ESBMC_assume(a >= -100 && a <= 100);
    __ESBMC_assume(b >= -100 && b <= 100);

    int mx = safe_max(a, b);
    __ESBMC_assert(mx >= a && mx >= b, "Max is >= both inputs");
    __ESBMC_assert(mx == a || mx == b, "Max is one of the inputs");

    int mn = safe_min(a, b);
    __ESBMC_assert(mn <= a && mn <= b, "Min is <= both inputs");

    int val = __ESBMC_nondet_int();
    __ESBMC_assume(val >= -200 && val <= 200);
    int clamped = clamp(val, -100, 100);
    __ESBMC_assert(clamped >= -100 && clamped <= 100, "Clamped in range");
}

/* ============================================
 * Example 5: Inheritance and Virtual Methods
 * ============================================ */

class Shape {
public:
    virtual int area() const = 0;
    virtual ~Shape() {}
};

class Rectangle : public Shape {
    int width, height;
public:
    Rectangle(int w, int h) : width(w), height(h) {
        __ESBMC_assert(w >= 0 && h >= 0, "Dimensions non-negative");
    }
    int area() const override {
        return width * height;
    }
    int get_width() const { return width; }
    int get_height() const { return height; }
};

class Square : public Rectangle {
public:
    Square(int side) : Rectangle(side, side) {}
};

void inheritance_example() {
    int w = __ESBMC_nondet_int();
    int h = __ESBMC_nondet_int();
    __ESBMC_assume(w >= 0 && w <= 100);
    __ESBMC_assume(h >= 0 && h <= 100);

    Rectangle rect(w, h);
    __ESBMC_assert(rect.area() == w * h, "Rectangle area correct");
    __ESBMC_assert(rect.area() >= 0, "Area is non-negative");

    int side = __ESBMC_nondet_int();
    __ESBMC_assume(side >= 0 && side <= 100);

    Square sq(side);
    __ESBMC_assert(sq.area() == side * side, "Square area correct");
    __ESBMC_assert(sq.get_width() == sq.get_height(), "Square has equal sides");
}

/* ============================================
 * Example 6: Operator Overloading
 * ============================================ */

class SafeInt {
    int val;

public:
    SafeInt(int v = 0) : val(v) {}

    SafeInt operator+(const SafeInt& other) const {
        // Prevent overflow by constraining inputs
        long long result = (long long)val + other.val;
        __ESBMC_assert(result >= -2147483648LL && result <= 2147483647LL,
                        "Addition does not overflow");
        return SafeInt((int)result);
    }

    bool operator==(const SafeInt& other) const {
        return val == other.val;
    }

    bool operator<(const SafeInt& other) const {
        return val < other.val;
    }

    int get() const { return val; }
};

void operator_overload_example() {
    int a_val = __ESBMC_nondet_int();
    int b_val = __ESBMC_nondet_int();
    __ESBMC_assume(a_val >= -1000 && a_val <= 1000);
    __ESBMC_assume(b_val >= -1000 && b_val <= 1000);

    SafeInt a(a_val);
    SafeInt b(b_val);
    SafeInt c = a + b;

    __ESBMC_assert(c.get() == a_val + b_val, "Addition correct");
}

/* ============================================
 * Example 7: Enum Class Safety
 * ============================================ */

enum class Color { Red, Green, Blue };

const char* color_name(Color c) {
    switch (c) {
        case Color::Red:   return "Red";
        case Color::Green: return "Green";
        case Color::Blue:  return "Blue";
    }
    __ESBMC_assert(false, "Unreachable: all enum values handled");
    return "";
}

void enum_example() {
    // Verify all enum values are handled
    __ESBMC_assert(color_name(Color::Red) != nullptr, "Red has a name");
    __ESBMC_assert(color_name(Color::Green) != nullptr, "Green has a name");
    __ESBMC_assert(color_name(Color::Blue) != nullptr, "Blue has a name");
}

int main() {
    class_invariant_example();
    vector_verification_example();
    raii_example();
    template_example();
    inheritance_example();
    operator_overload_example();
    enum_example();

    return 0;
}
