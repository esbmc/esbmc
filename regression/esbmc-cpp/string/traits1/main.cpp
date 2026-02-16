#include <string>
#include <iostream>
#include <cassert>
#include <cstring>

// Test case demonstrating char_traits usage in C++
// Save as: char_traits_test.cpp
// Compile with: clang++ -std=c++11 char_traits_test.cpp -o char_traits_test

void test_basic_char_traits_operations() {
    std::cout << "=== Testing basic char_traits operations ===" << std::endl;
    
    // Test char_traits<char> directly
    typedef std::char_traits<char> traits;
    
    // Test character comparison
    char c1 = 'a', c2 = 'b', c3 = 'a';
    
    std::cout << "char_traits::eq('a', 'a'): " << traits::eq(c1, c3) << std::endl;
    std::cout << "char_traits::eq('a', 'b'): " << traits::eq(c1, c2) << std::endl;
    std::cout << "char_traits::lt('a', 'b'): " << traits::lt(c1, c2) << std::endl;
    std::cout << "char_traits::lt('b', 'a'): " << traits::lt(c2, c1) << std::endl;
    
    // Test string length
    const char* test_str = "Hello, World!";
    size_t len = traits::length(test_str);
    std::cout << "char_traits::length(\"" << test_str << "\"): " << len << std::endl;
    
    // Test string comparison
    const char* str1 = "hello";
    const char* str2 = "world";
    const char* str3 = "hello";
    
    int cmp1 = traits::compare(str1, str2, 5);
    int cmp2 = traits::compare(str1, str3, 5);
    
    std::cout << "char_traits::compare(\"hello\", \"world\", 5): " << cmp1 << std::endl;
    std::cout << "char_traits::compare(\"hello\", \"hello\", 5): " << cmp2 << std::endl;
    
    // Test character search
    const char* found = traits::find(test_str, strlen(test_str), 'W');
    if (found) {
        std::cout << "char_traits::find found 'W' at position: " << (found - test_str) << std::endl;
    }
    
    // Test character assignment and copying
    char buffer[20];
    traits::copy(buffer, "Test copy", 9);
    buffer[9] = '\0';
    std::cout << "char_traits::copy result: \"" << buffer << "\"" << std::endl;
    
    // Test character assignment (fill)
    char fill_buffer[10];
    traits::assign(fill_buffer, 9, 'X');
    fill_buffer[9] = '\0';
    std::cout << "char_traits::assign(9, 'X') result: \"" << fill_buffer << "\"" << std::endl;
    
    // Test move operation
    char move_buffer[20] = "Original text";
    traits::move(move_buffer + 2, move_buffer, 8); // Overlapping move
    move_buffer[10] = '\0';
    std::cout << "char_traits::move result: \"" << move_buffer << "\"" << std::endl;
}

void test_char_traits_with_string_operations() {
    std::cout << "\n=== Testing char_traits with string operations ===" << std::endl;
    
    // Create strings that internally use char_traits
    std::string s1("Hello");
    std::string s2("World");
    std::string s3("Hello");
    
    // These operations internally use char_traits methods
    std::cout << "String comparison (s1 == s2): " << (s1 == s2) << std::endl;
    std::cout << "String comparison (s1 == s3): " << (s1 == s3) << std::endl;
    std::cout << "String comparison (s1 < s2): " << (s1 < s2) << std::endl;
    
    // String concatenation (uses char_traits for copying)
    std::string concatenated = s1 + " " + s2;
    std::cout << "Concatenated string: \"" << concatenated << "\"" << std::endl;
    
    // String search operations (use char_traits::find internally)
    size_t pos = concatenated.find('W');
    if (pos != std::string::npos) {
        std::cout << "Found 'W' at position: " << pos << std::endl;
    }
    
    // Character access (uses char_traits for validation)
    if (!concatenated.empty()) {
        std::cout << "First character: '" << concatenated[0] << "'" << std::endl;
        std::cout << "Last character: '" << concatenated.back() << "'" << std::endl;
    }
    
    // Substring operations (use char_traits for copying)
    std::string sub = concatenated.substr(6, 5);
    std::cout << "Substring(6, 5): \"" << sub << "\"" << std::endl;
}

void test_char_traits_special_values() {
    std::cout << "\n=== Testing char_traits special values ===" << std::endl;
    
    typedef std::char_traits<char> traits;
    
    // Test EOF and character conversion
    auto eof_val = traits::eof();
    std::cout << "char_traits::eof(): " << eof_val << std::endl;
    
    char test_char = 'A';
    auto int_val = traits::to_int_type(test_char);
    auto char_val = traits::to_char_type(int_val);
    
    std::cout << "to_int_type('A'): " << int_val << std::endl;
    std::cout << "to_char_type(back to char): '" << char_val << "'" << std::endl;
    
    // Test not_eof
    auto not_eof_result = traits::not_eof(eof_val);
    std::cout << "not_eof(eof()): " << not_eof_result << std::endl;
    
    auto not_eof_normal = traits::not_eof(int_val);
    std::cout << "not_eof(normal char): " << not_eof_normal << std::endl;
    
    // Test eq_int_type
    std::cout << "eq_int_type(eof, eof): " << traits::eq_int_type(eof_val, eof_val) << std::endl;
    std::cout << "eq_int_type('A', 'A'): " << traits::eq_int_type(int_val, int_val) << std::endl;
}

void test_custom_char_traits_usage() {
    std::cout << "\n=== Testing custom char_traits usage ===" << std::endl;
    
    // Demonstrate how char_traits can be customized
    // This shows what an operational model needs to support
    
    // Create a basic_string with explicit char_traits specification
    std::basic_string<char, std::char_traits<char>, std::allocator<char>> explicit_string("Test");
    
    std::cout << "Explicit basic_string: \"" << explicit_string << "\"" << std::endl;
    std::cout << "Length: " << explicit_string.length() << std::endl;
    std::cout << "Empty: " << explicit_string.empty() << std::endl;
    
    // Show that std::string is just a typedef
    std::string normal_string("Same thing");
    std::cout << "Normal string: \"" << normal_string << "\"" << std::endl;
    
    // Test that they're compatible
    explicit_string = normal_string;
    std::cout << "After assignment: \"" << explicit_string << "\"" << std::endl;
    
    // Demonstrate assignment and concatenation operations
    explicit_string += " appended";
    std::cout << "After append: \"" << explicit_string << "\"" << std::endl;
}

void test_char_traits_edge_cases() {
    std::cout << "\n=== Testing char_traits edge cases ===" << std::endl;
    
    typedef std::char_traits<char> traits;
    
    // Test with empty strings
    const char* empty_str = "";
    std::cout << "Length of empty string: " << traits::length(empty_str) << std::endl;
    
    // Test with null character
    char null_char = '\0';
    std::cout << "to_int_type('\\0'): " << traits::to_int_type(null_char) << std::endl;
    
    // Test comparison with identical pointers
    const char* same_ptr = "test";
    int self_cmp = traits::compare(same_ptr, same_ptr, 4);
    std::cout << "Compare same pointer to itself: " << self_cmp << std::endl;
    
    // Test find with character not present
    const char* not_found = traits::find("hello", 5, 'z');
    std::cout << "Find non-existent character: " << (not_found ? "found" : "not found") << std::endl;
    
    // Test zero-length operations
    char buffer[10] = "unchanged";
    traits::copy(buffer, "test", 0);  // Copy 0 characters
    std::cout << "After zero-length copy: \"" << buffer << "\"" << std::endl;
}

int main() {
    std::cout << "C++ char_traits Comprehensive Test\n";
    std::cout << "==================================\n" << std::endl;
    
    try {
        test_basic_char_traits_operations();
        test_char_traits_with_string_operations();
        test_char_traits_special_values();
        test_custom_char_traits_usage();
        test_char_traits_edge_cases();
        
        std::cout << "\n=== All tests completed successfully ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

