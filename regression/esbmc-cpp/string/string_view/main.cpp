#include <string_view>
#include <cassert>
#include <cstring>
#include <iostream>

void test_string_view_copy() {
    // Test 1: Basic copy operation
    {
        std::string_view sv("Hello World");
        char buffer[20] = {0};
        
        size_t copied = sv.copy(buffer, 5);
        
        assert(copied == 5);
        assert(strncmp(buffer, "Hello", 5) == 0);
        assert(buffer[5] == '\0');  // Ensure null termination from initialization
        
        std::cout << "Test 1 passed: Basic copy" << std::endl;
    }
    
    // Test 2: Copy with position offset
    {
        std::string_view sv("Hello World");
        char buffer[20] = {0};
        
        size_t copied = sv.copy(buffer, 5, 6);  // Copy "World" starting at position 6
        
        assert(copied == 5);
        assert(strncmp(buffer, "World", 5) == 0);
        
        std::cout << "Test 2 passed: Copy with position offset" << std::endl;
    }
    
    // Test 3: Copy more than available (should copy only what's available)
    {
        std::string_view sv("Hi");
        char buffer[20] = {0};
        
        size_t copied = sv.copy(buffer, 10);  // Request 10, but only 2 available
        
        assert(copied == 2);
        assert(strncmp(buffer, "Hi", 2) == 0);
        
        std::cout << "Test 3 passed: Copy more than available" << std::endl;
    }
    
    // Test 4: Copy with position near end
    {
        std::string_view sv("Hello");
        char buffer[20] = {0};
        
        size_t copied = sv.copy(buffer, 10, 3);  // Start at position 3, only "lo" available
        
        assert(copied == 2);
        assert(strncmp(buffer, "lo", 2) == 0);
        
        std::cout << "Test 4 passed: Copy with position near end" << std::endl;
    }
    
    // Test 5: Copy zero characters
    {
        std::string_view sv("Hello");
        char buffer[20] = {0};
        buffer[0] = 'X';  // Set marker
        
        size_t copied = sv.copy(buffer, 0);
        
        assert(copied == 0);
        assert(buffer[0] == 'X');  // Should be unchanged
        
        std::cout << "Test 5 passed: Copy zero characters" << std::endl;
    }
    
    // Test 6: Position at end of string
    {
        std::string_view sv("Hello");
        char buffer[20] = {0};
        
        size_t copied = sv.copy(buffer, 5, 5);  // Start at position 5 (end of string)
        
        assert(copied == 0);
        
        std::cout << "Test 6 passed: Position at end of string" << std::endl;
    }
    
    // Test 7: Copy entire string
    {
        std::string_view sv("Test123");
        char buffer[20] = {0};
        
        size_t copied = sv.copy(buffer, 100);  // Request more than length
        
        assert(copied == 7);  // Length of "Test123"
        assert(strncmp(buffer, "Test123", 7) == 0);
        
        std::cout << "Test 7 passed: Copy entire string" << std::endl;
    }
    
    // Test 8: Empty string_view
    {
        std::string_view sv("");
        char buffer[20] = {0};
        buffer[0] = 'X';  // Set marker
        
        size_t copied = sv.copy(buffer, 5);
        
        assert(copied == 0);
        assert(buffer[0] == 'X');  // Should be unchanged
        
        std::cout << "Test 8 passed: Empty string_view" << std::endl;
    }
    
    // Test 9: Copy exact length
    {
        std::string_view sv("ABC");
        char buffer[20] = {0};
        
        size_t copied = sv.copy(buffer, 3);
        
        assert(copied == 3);
        assert(strncmp(buffer, "ABC", 3) == 0);
        
        std::cout << "Test 9 passed: Copy exact length" << std::endl;
    }
    
    // Test 10: Verify buffer contents after partial copy
    {
        std::string_view sv("ABCDEF");
        char buffer[20];
        memset(buffer, 'Z', sizeof(buffer));  // Fill with 'Z'
        
        size_t copied = sv.copy(buffer, 3, 1);  // Copy "BCD" starting at position 1
        
        assert(copied == 3);
        assert(buffer[0] == 'B');
        assert(buffer[1] == 'C');
        assert(buffer[2] == 'D');
        assert(buffer[3] == 'Z');  // Should be unchanged
        
        std::cout << "Test 10 passed: Verify buffer contents after partial copy" << std::endl;
    }
}

int main() {
    std::cout << "Testing string_view::copy method..." << std::endl;
    
    test_string_view_copy();
    
    std::cout << "\nAll tests passed! ✅" << std::endl;
    std::cout << "The string_view::copy method works correctly and returns the proper count." << std::endl;
    
    return 0;
}

