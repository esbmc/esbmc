#pragma once

// RM: I really thought about making this type-safe with visitor pattern. But I choose simplicity for now.

enum class AssertionType
{
  MISALIGNED_MEMORY_ACCESS,
  USER_ASSERTION,
  ATOMICITY_CHECK,
  BOUNDS_CHECK,
  DATA_RACE_CHECK,
  DEADLOCK_CHECK,
  DIV_BY_ZERO_CHECK,
  LOCK_ORDER_CHECK,
  NAN_CHECK,
  OVERFLOW_CHECK,
  POINTER_CHECK,
  STRUCT_FIELD_CHECK,
  UB_SHIFT_CHECK,
  UNLIMITED_SCANF_CHECK,
  UNSIGNED_OVERFLOW_CHECK,
  VLA_CHECK,
  INTRINSIC,  
  OTHER
};

