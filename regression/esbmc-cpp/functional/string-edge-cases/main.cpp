#include <cassert>
#include <functional>
#include <string>

int main() {
    std::hash<std::string> string_hasher;
    std::hash<const char*> cstr_hasher;
    
    // Test various string edge cases
    std::string empty_str = "";
    std::string single_char = "a";
    std::string null_char = std::string(1, '\0');
    std::string spaces = "   ";
    std::string newlines = "\n\r\t";
    std::string very_long = std::string(1000, 'x');
    std::string unicode = "héllo wørld";
    std::string special_chars = "!@#$%^&*()";
    std::string numbers = "1234567890";
    std::string mixed = "Hello123!";
    
    // Get hashes
    std::size_t hash_empty = string_hasher(empty_str);
    std::size_t hash_single = string_hasher(single_char);
    std::size_t hash_null = string_hasher(null_char);
    std::size_t hash_spaces = string_hasher(spaces);
    std::size_t hash_newlines = string_hasher(newlines);
    std::size_t hash_long = string_hasher(very_long);
    std::size_t hash_unicode = string_hasher(unicode);
    std::size_t hash_special = string_hasher(special_chars);
    std::size_t hash_numbers = string_hasher(numbers);
    std::size_t hash_mixed = string_hasher(mixed);
    
    // Test determinism
    assert(string_hasher("") == hash_empty);
    assert(string_hasher("a") == hash_single);
    assert(string_hasher("   ") == hash_spaces);
    assert(string_hasher(very_long) == hash_long);
    assert(string_hasher("1234567890") == hash_numbers);
    
    // Test that different strings produce different hashes
    // (not guaranteed by standard, but likely for these cases)
    assert(hash_empty != hash_single);
    assert(hash_single != hash_spaces);
    assert(hash_numbers != hash_mixed);
    
    // All should be valid
    assert(hash_empty >= 0);
    assert(hash_long >= 0);
    assert(hash_unicode >= 0);
    
    return 0;
}

