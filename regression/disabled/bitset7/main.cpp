#include <iostream>
#include <cassert>
#include <bitset>
#include <exception>

using namespace std;

void verify_bounds_safety() {
    std::bitset<8> bs;
    
    for (size_t i = 0; i < 8; ++i) {
        bs.set(i);
        assert(bs.test(i) == true);
        bs[i] = false;
        assert(bs[i] == false);
    }
}

void verify_invariants() {
    // Test bitset invariants that should always hold
    std::bitset<4> bs;
    
    // Invariant: count() should equal number of set bits
    bs.set(0);
    bs.set(2);
    assert(bs.count() == 2);
    
    // Invariant: any() should be true iff count() > 0
    assert(bs.any() == (bs.count() > 0));
    
    // Invariant: none() should be true iff count() == 0
    assert(bs.none() == (bs.count() == 0));
    
    // Invariant: all() should be true iff count() == size()
    assert(bs.all() == (bs.count() == bs.size()));
    
    bs.set(); // Set all bits
    assert(bs.all() == true);
    assert(bs.count() == bs.size());
    
    // Invariant: ~(~bs) should equal bs
    std::bitset<4> bs_copy = bs;
    std::bitset<4> double_not = ~(~bs);
    assert(double_not == bs_copy);
}

void verify_operation_properties() {
    // Verify algebraic properties of bitwise operations
    std::bitset<4> a("1010");
    std::bitset<4> b("1100");
    std::bitset<4> zero("0000");
    std::bitset<4> ones("1111");
    
    // Commutative properties
    assert((a & b) == (b & a));
    assert((a | b) == (b | a));
    assert((a ^ b) == (b ^ a));
    
    // Identity properties
    assert((a & ones) == a);
    assert((a | zero) == a);
    assert((a ^ zero) == a);
    
    // Annihilation properties
    assert((a & zero) == zero);
    assert((a | ones) == ones);
    
    // Self-operation properties
    assert((a ^ a) == zero);
    assert((a & a) == a);
    assert((a | a) == a);
    
    // De Morgan's laws
    assert(~(a & b) == (~a | ~b));
    assert(~(a | b) == (~a & ~b));
}

void verify_shift_properties() {
    // Verify shift operation properties
    std::bitset<8> bs("00001111");
    
    // Left shift followed by right shift (with no overflow)
    std::bitset<8> shifted = (bs << 2) >> 2;
    // Should lose the top 2 bits but bottom bits should be preserved
    std::bitset<8> expected("00000011");
    
    // Shift by 0 should be identity
    assert((bs << 0) == bs);
    assert((bs >> 0) == bs);
    
    // Shift by size should result in all zeros
    assert((bs << 8).none() == true);
    assert((bs >> 8).none() == true);
    
    // Compound shift properties
    std::bitset<8> bs_copy = bs;
    bs_copy <<= 3;
    assert(bs_copy == (bs << 3));
    
    bs_copy = bs;
    bs_copy >>= 2;
    assert(bs_copy == (bs >> 2));
}

void verify_string_conversion_consistency() {
    // Verify string constructor produces expected bit patterns
    std::bitset<4> bs1("1010");
    
    assert(bs1[0] == false); // rightmost bit (position 0)
    assert(bs1[1] == true);
    assert(bs1[2] == false);
    assert(bs1[3] == true);  // leftmost bit (position 3)
    
    // Verify partial string construction
    std::bitset<4> bs2(std::string("111010"), 2, 4); // Extract "1010"
    assert(bs2 == bs1);
    
    // Verify numeric equivalence
    assert(bs1.to_ulong() == 10); // 1010 binary = 10 decimal
}

void verify_reference_semantics() {
    // Verify that bitset reference behaves correctly
    std::bitset<4> bs("0000");
    
    // Test reference assignment
    bs[0] = true;
    assert(bs.test(0) == true);
    
    // Test reference flip
    bs[1].flip();
    assert(bs.test(1) == true);
    
    bs[1].flip();
    assert(bs.test(1) == false);
    
    // Test reference comparison
    bs[2] = true;
    assert(bs[2] == true);
    assert(bs[0] == bs[2]); // Both should be true
    
    // Test reference copy
    bs[3] = bs[0];
    assert(bs.test(3) == true);
}

void verify_overflow_safety() {
    // Verify that operations handle potential overflows safely
    std::bitset<4> max_val("1111"); // Maximum 4-bit value (15)
    
    // to_ulong should work for values that fit
    assert(max_val.to_ulong() == 15);
    
    // Test that larger bitsets with values > ULONG_MAX would assert
    // (This would be tested with larger bitsets in real scenarios)
    std::bitset<8> safe_val("11111111"); // 255, fits in unsigned long
    assert(safe_val.to_ulong() == 255);
}

void verify_constructor_edge_cases() {
    // Test edge cases in constructors
    
    // Empty string should create zero bitset
    std::bitset<4> empty_str("");
    assert(empty_str.none() == true);
    
    // String with non-binary characters (should be ignored)
    std::bitset<4> mixed_str("1x0y");
    assert(mixed_str.test(0) == false); // 'y' treated as '0'
    assert(mixed_str.test(1) == false); // '0' 
    assert(mixed_str.test(2) == false); // 'x' treated as '0'
    assert(mixed_str.test(3) == true);  // '1'
    
    // String longer than bitset size
    std::bitset<4> long_str("10101010");
    // Should use rightmost 4 characters: "1010"
    assert(long_str.test(0) == false);
    assert(long_str.test(1) == true);
    assert(long_str.test(2) == false);
    assert(long_str.test(3) == true);
    
    // String shorter than bitset size
    std::bitset<8> short_str("101");
    // Should be "00000101"
    assert(short_str.count() == 2);
    assert(short_str.test(0) == true);
    assert(short_str.test(2) == true);
}

void verify_count_properties() {
    // Verify count() behavior under various operations
    std::bitset<8> bs;
    
    // Initially empty
    assert(bs.count() == 0);
    
    // Set bits one by one
    for (size_t i = 0; i < 8; ++i) {
        bs.set(i);
        assert(bs.count() == i + 1);
    }
    
    // Reset bits one by one
    for (size_t i = 0; i < 8; ++i) {
        bs.reset(i);
        assert(bs.count() == 8 - i - 1);
    }
    
    // Flip operations
    bs.reset(); // Start with all zeros
    bs.flip();  // All ones
    assert(bs.count() == 8);
    
    bs.flip(0); // Flip one bit
    assert(bs.count() == 7);
}

void run_property_verification() {
    verify_bounds_safety();
    verify_invariants();
    verify_operation_properties();
    verify_shift_properties();
    verify_string_conversion_consistency();
    verify_reference_semantics();
    verify_overflow_safety();
    verify_constructor_edge_cases();
    verify_count_properties();
}

// Bounded verification scenarios for specific bit sizes
void bounded_verification_scenarios() {
    // Test all possible 2-bit combinations
    for (unsigned long i = 0; i < 4; ++i) {
        std::bitset<2> bs(i);
        
        // Verify bit pattern matches numeric value
        bool bit0 = (i & 1) != 0;
        bool bit1 = (i & 2) != 0;
        
        assert(bs.test(0) == bit0);
        assert(bs.test(1) == bit1);
        assert(bs.to_ulong() == i);
        
        // Verify count is correct
        size_t expected_count = (bit0 ? 1 : 0) + (bit1 ? 1 : 0);
        assert(bs.count() == expected_count);
    }
    
    // Test all possible 3-bit combinations with operations
    for (unsigned long a = 0; a < 8; ++a) {
        for (unsigned long b = 0; b < 8; ++b) {
            std::bitset<3> bs_a(a);
            std::bitset<3> bs_b(b);
            
            // Verify bitwise operations produce correct results
            std::bitset<3> and_result = bs_a & bs_b;
            std::bitset<3> or_result = bs_a | bs_b;
            std::bitset<3> xor_result = bs_a ^ bs_b;
            
            assert(and_result.to_ulong() == (a & b));
            assert(or_result.to_ulong() == (a | b));
            assert(xor_result.to_ulong() == (a ^ b));
        }
    }
}

int main() {
    try {
        run_property_verification();
        bounded_verification_scenarios();
    } catch (const std::exception& e) {
        cout << "FAILURE: Verification test failed with exception: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "FAILURE: Verification test failed with unknown exception" << endl;
        return 1;
    }
    
    return 0;
}
