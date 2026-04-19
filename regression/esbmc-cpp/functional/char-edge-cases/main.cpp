#include <cassert>
#include <functional>

int main() {
    std::hash<char> char_hasher;
    std::hash<unsigned char> uchar_hasher;
    std::hash<wchar_t> wchar_hasher;
    
    // Test ASCII control characters
    std::size_t hash_null = char_hasher('\0');
    std::size_t hash_tab = char_hasher('\t');
    std::size_t hash_newline = char_hasher('\n');
    std::size_t hash_cr = char_hasher('\r');
    std::size_t hash_esc = char_hasher('\033');
    std::size_t hash_del = char_hasher('\177');
    
    // Test printable characters
    std::size_t hash_space = char_hasher(' ');
    std::size_t hash_exclaim = char_hasher('!');
    std::size_t hash_tilde = char_hasher('~');
    std::size_t hash_a = char_hasher('a');
    std::size_t hash_A = char_hasher('A');
    std::size_t hash_0 = char_hasher('0');
    std::size_t hash_9 = char_hasher('9');
    
    // Test unsigned char boundaries
    std::size_t hash_uchar_0 = uchar_hasher(0);
    std::size_t hash_uchar_255 = uchar_hasher(255);
    
    // Test wide characters
    std::size_t hash_wchar_0 = wchar_hasher(L'\0');
    std::size_t hash_wchar_a = wchar_hasher(L'a');
    std::size_t hash_wchar_unicode = wchar_hasher(L'é');
    
    // Test determinism
    assert(char_hasher('\0') == hash_null);
    assert(char_hasher('\n') == hash_newline);
    assert(char_hasher('a') == hash_a);
    assert(char_hasher('A') == hash_A);
    assert(uchar_hasher(255) == hash_uchar_255);
    assert(wchar_hasher(L'é') == hash_wchar_unicode);
    
    // Test that different characters give different hashes
    assert(hash_a != hash_A);  // Case sensitivity
    assert(hash_0 != hash_9);  // Different digits
    assert(hash_space != hash_tab);  // Different whitespace
    
    // All should be valid
    assert(hash_null >= 0);
    assert(hash_wchar_unicode >= 0);  // ← Fixed: was hash_unicode
    assert(hash_uchar_255 >= 0);
    
    return 0;
}
