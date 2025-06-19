#include <cassert>
#include <functional>
#include <cmath>
#include <cfloat>

int main() {
    std::hash<float> float_hasher;
    std::hash<double> double_hasher;
    
    // Test special float values
    std::size_t hash_nan = float_hasher(nan(""));  // Use global nan instead of std::nan
    std::size_t hash_zero = float_hasher(0.0f);
    std::size_t hash_neg_zero = float_hasher(-0.0f);
    std::size_t hash_min = float_hasher(FLT_MIN);
    std::size_t hash_max = float_hasher(FLT_MAX);
    std::size_t hash_epsilon = float_hasher(FLT_EPSILON);
    
    // Test subnormal numbers
    std::size_t hash_denorm_min = float_hasher(FLT_TRUE_MIN);
    
    // Test double precision values
    std::size_t hash_double_max = double_hasher(DBL_MAX);
    std::size_t hash_double_min = double_hasher(DBL_MIN);
    std::size_t hash_double_epsilon = double_hasher(DBL_EPSILON);
    
    // Test very small and very large values
    std::size_t hash_tiny = float_hasher(1e-30f);
    std::size_t hash_huge = float_hasher(1e30f);
    
    // Determinism tests for special values
    assert(float_hasher(0.0f) == hash_zero);
    assert(float_hasher(FLT_MAX) == hash_max);
    assert(double_hasher(DBL_MAX) == hash_double_max);
    
    // NaN special case - NaN != NaN, but hash should be deterministic
    assert(float_hasher(nan("")) == hash_nan);
    
    // Test that +0.0 and -0.0 might have same or different hashes
    // (implementation defined, but should be deterministic)
    assert(float_hasher(0.0f) == hash_zero);
    assert(float_hasher(-0.0f) == hash_neg_zero);
    
    // Test determinism for various values
    assert(float_hasher(1e-30f) == hash_tiny);
    assert(float_hasher(1e30f) == hash_huge);
    assert(float_hasher(FLT_EPSILON) == hash_epsilon);
    
    // All should be valid (size_t is always >= 0)
    assert(hash_nan >= 0);
    assert(hash_max >= 0);
    assert(hash_double_max >= 0);
    assert(hash_tiny >= 0);
    assert(hash_huge >= 0);
    
    return 0;
}
