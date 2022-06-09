/*******************************************************************\
 Module: GOTO Assert Mode
 Author: Rafael Sá Menezes
 Date: June 2022

 Description: The assert mode will categorize every assertion that
              ESBMC does. This needs to works with primitive integers
              as user will use that to comunicate with the symex
\*******************************************************************/


namespace goto_assertions 
{
/**
 * Assertions should belong to a specific category
 */
enum goto_assertion_mode {
  USER = 1, // __ESBMC_assert, assert
  POINTER_SAFETY = 2, // leaks, double-free, segmentation fault, dereference invalid address
  ARRAY_SAFETY = 4, // out-of-bounds
  ARITHMETIC_SAFETY = 8, // arithmetic overflows/underflows, division by zero, NaN
  OTHER = 16 // Default for every other assertion (TODO: will remove this)
};

constexpr goto_assertion_mode ALL_MODES = (goto_assertion_mode) 0xFF;

inline bool is_mode_enabled(const goto_assertion_mode &assertions, char mode) {
    return (mode & assertions) != 0;
};


} // namespace goto_assertions 