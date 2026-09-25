#ifndef PYTHON_FRONTEND_PYTHON_INT_OVERFLOW_H
#define PYTHON_FRONTEND_PYTHON_INT_OVERFLOW_H

#include <stdexcept>

// Marker exception for Python int overflow diagnostics (issue #4642).
//
// Derived from std::runtime_error so existing `catch (const std::exception &)`
// sites in the frontend pipeline continue to receive it for orderly shutdown.
// The point of the subclass is to mark these diagnostics so f-string and
// similar fallback handlers can re-throw rather than downgrade to a warning
// (which would silently consume the bignum and let an unrelated assertion
// "succeed").
class python_int_overflow_excp : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

#endif
