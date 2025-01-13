#include <iostream>
#include <cctype> // For isblank
#include <cassert> // For assert

void test_isblank() {
    // Test with blank characters
    assert(isblank(' ') != 0);  // Space is blank
    assert(isblank('\t') != 0); // Tab is blank

    // Test with non-blank characters
    assert(isblank('a') == 0);  // Letter 'a' is not blank
    assert(isblank('1') == 0);  // Digit '1' is not blank
    assert(isblank('\n') == 0); // Newline is not blank
    assert(isblank('\v') == 0); // Vertical tab is not blank
    assert(isblank('\0') == 0); // Null character is not blank
    assert(isblank('*') == 0);  // Asterisk is not blank

    // Test with extended ASCII characters
    assert(isblank('\xA0') == 0); // Non-breaking space in extended ASCII

    // Test with EOF
    assert(isblank(EOF) == 0); // EOF should not be considered blank

    std::cout << "All tests passed for isblank()" << std::endl;
}

int main() {
    test_isblank();
    return 0;
}

