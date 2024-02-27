#ifndef BITWISE_BOUNDS_H_INCLUDED
#define BITWISE_BOUNDS_H_INCLUDED

#define UINT_X unsigned /// replace this with appropriate unsigned integer type
#define INT_X int       /// replace this with appropriate signed integer type

/**
 * Adaptation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * for unsigned integers of any width, by scanning from the least significant bit.
 * @author Edoardo Manino.
 * @brief Maximum value of x | y given x in [a,b] and y in [c,d]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
UINT_X unsigned_max_or(UINT_X a, UINT_X b, UINT_X c, UINT_X d)
{
  UINT_X extra_bits = 0;

  // scan all the bits that are active in both upper bounds
  UINT_X cand_bits = b & d;
  while (cand_bits != 0)
  {
    // isolate the least significant candidate bit
    // e.g. cand_bits = 0b01101010 yields least_bit = 0b00000010
    UINT_X least_bit = cand_bits & (-cand_bits);

    // reduce b and d by removing least_bit and activating all the less significant ones
    // e.g. b = 0b00101100 and least_bit = 0b00001000 yields alternate_b = 0b00100111
    UINT_X alternate_b = (b - least_bit) | (least_bit - 1);
    UINT_X alternate_d = (d - least_bit) | (least_bit - 1);

    // stop only if both alternate values are outside of the valid interval
    if (alternate_b < a && alternate_d < c)
      break;

    // collect the extra activated bits
    // e.g. least_bit = 0b00001000 yields extra_bits |= 0b00000111
    extra_bits |= (least_bit - 1);

    // remove the least significant bit from the candidates until none is left
    cand_bits &= (cand_bits - 1);
  }

  // the maximum value is the disjunction of both upper bounds and the extra bits
  return b | d | extra_bits;
}

/**
 * Adaptation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * for unsigned integers of any width, by scanning from the least significant bit.
 * @author Edoardo Manino.
 * @brief Minimum value of x | y given x in [a,b] and y in [c,d]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
UINT_X unsigned_min_or(UINT_X a, UINT_X b, UINT_X c, UINT_X d)
{
  UINT_X best_a = a;
  UINT_X best_c = c;

  // scan all the bits that are already active in the other lower bound
  UINT_X cand_bits = ~a & c;
  while (cand_bits != 0)
  {
    // isolate the least significant candidate bit
    // e.g. cand_bits = 0b01101010 yields least_bit = 0b00000010
    UINT_X least_bit = cand_bits & (-cand_bits);

    // increase a by adding least_bit and de-activating all the less significant ones
    // e.g. a = 0b00101100 and least_bit = 0b00010000 yields alternate_a = 0b00110000
    UINT_X alternate_a = (a | least_bit) & (-least_bit);

    // stop if the alternate value is outside of the valid interval
    if (alternate_a > b)
      break;

    // keep track of the best value of a found so far
    best_a = alternate_a;

    // remove the least significant bit from the candidates until none is left
    cand_bits &= (cand_bits - 1);
  }

  // repeat the same procedure for the interval y in [c,d]
  cand_bits = a & ~c;
  while (cand_bits != 0)
  {
    UINT_X least_bit = cand_bits & (-cand_bits);
    UINT_X alternate_c = (c | least_bit) & (-least_bit);
    if (alternate_c > d)
      break;
    best_c = alternate_c;
    cand_bits &= (cand_bits - 1);
  }

  // compute two separate bounds with the modified a and c
  // note: since best_a >= a and best_c >= c,
  // using best_a | best_c could give unsound results
  UINT_X min_1 = best_a | c;
  UINT_X min_2 = a | best_c;

  // return the lowest bound
  return (min_1 < min_2) ? min_1 : min_2;
}

/**
 * Implementation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Maximum value of x & y given x in [a,b] and y in [c,d]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
UINT_X unsigned_max_and(UINT_X a, UINT_X b, UINT_X c, UINT_X d)
{
  return ~unsigned_min_or(~b, ~a, ~d, ~c);
}

/**
 * Implementation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Minimum value of x & y given x in [a,b] and y in [c,d]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
UINT_X unsigned_min_and(UINT_X a, UINT_X b, UINT_X c, UINT_X d)
{
  return ~unsigned_max_or(~b, ~a, ~d, ~c);
}

/**
 * Implementation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Maximum value of x ^ y given x in [a,b] and y in [c,d]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
UINT_X unsigned_max_xor(UINT_X a, UINT_X b, UINT_X c, UINT_X d)
{
  return unsigned_max_or(
    0, unsigned_max_and(a, b, ~d, ~c), 0, unsigned_max_and(~b, ~a, c, d));
}

/**
 * Implementation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Minimum value of x ^ y given x in [a,b] and y in [c,d]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
UINT_X unsigned_min_xor(UINT_X a, UINT_X b, UINT_X c, UINT_X d)
{
  return unsigned_min_and(a, b, ~d, ~c) | unsigned_min_and(~b, ~a, c, d);
}

/**
 * Implementation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Maximum value of ~x given x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 */
UINT_X unsigned_max_not(UINT_X a, UINT_X)
{
  return ~a;
}

/**
 * Implementation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Minimum value of ~x given x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 */
UINT_X unsigned_min_not(UINT_X, UINT_X b)
{
  return ~b;
}

/**
 * Upper bound on interval left shift
 * @author Edoardo Manino.
 * @brief Maximum value of x << y given x in [a,b] and y in [c,d]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
UINT_X unsigned_max_left_shift(UINT_X, UINT_X b, UINT_X, UINT_X d)
{
  return b << d;
}

/**
 * Lower bound on interval left shift
 * @author Edoardo Manino.
 * @brief Minimum value of x << y given x in [a,b] and y in [c,d]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
UINT_X unsigned_min_left_shift(UINT_X a, UINT_X, UINT_X c, UINT_X)
{
  return a << c;
}

/**
 * Upper bound on interval right shift
 * @author Edoardo Manino.
 * @brief Maximum value of x >> y given x in [a,b] and y in [c,d]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
UINT_X unsigned_max_right_shift(UINT_X, UINT_X b, UINT_X c, UINT_X)
{
  return b >> c;
}

/**
 * Lower bound on interval right shift
 * @author Edoardo Manino.
 * @brief Minimum value of x >> y given x in [a,b] and y in [c,d]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
UINT_X unsigned_min_right_shift(UINT_X a, UINT_X, UINT_X, UINT_X d)
{
  return a >> d;
}

/**
 * Upper bound on unsigned to unsigned interval truncation
 * @author Edoardo Manino.
 * @brief Maximum value of unsigned truncate(x, n) given n bits and unsigned x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param n Number of bits of the result
 */
UINT_X unsigned_2_unsigned_max_truncate(UINT_X a, UINT_X b, UINT_X n)
{
  // set the n least significant bits to one
  UINT_X sign_bit = 1 << (n - 1);
  UINT_X mask = sign_bit | (sign_bit - 1);

  // compute the maximum value of x & mask
  return unsigned_max_and(a, b, mask, mask);
}

/**
 * Lower bound on unsigned to unsigned interval truncation
 * @author Edoardo Manino.
 * @brief Minimum value of unsigned truncate(x, n) given n bits and unsigned x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param n Number of bits of the result
 */
UINT_X unsigned_2_unsigned_min_truncate(UINT_X a, UINT_X b, UINT_X n)
{
  // set the n least significant bits to one
  UINT_X sign_bit = 1 << (n - 1);
  UINT_X mask = sign_bit | (sign_bit - 1);

  // compute the minimum value of x & mask
  return unsigned_min_and(a, b, mask, mask);
}

/**
 * Upper bound on signed to unsigned interval truncation
 * @author Edoardo Manino.
 * @brief Maximum value of unsigned truncate(x, n) given n bits and signed x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param n Number of bits of the result
 */
UINT_X signed_2_unsigned_max_truncate(INT_X a, INT_X b, UINT_X n)
{
  // if the input interval contains zero, return truncated -1
  if (a < 0 && b >= 0)
    return (1 << n) - 1;

  // otherwise interpret [a,b] as an unsigned interval
  else
    return unsigned_2_unsigned_max_truncate(a, b, n);
}

/**
 * Lower bound on signed to unsigned interval truncation
 * @author Edoardo Manino.
 * @brief Minimum value of unsigned truncate(x, n) given n bits and signed x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param n Number of bits of the result
 */
UINT_X signed_2_unsigned_min_truncate(INT_X a, INT_X b, UINT_X n)
{
  // if the input interval contains zero, return zero
  if (a < 0 && b >= 0)
    return 0;

  // otherwise interpret [a,b] as an unsigned interval
  else
    return unsigned_2_unsigned_min_truncate(a, b, n);
}

/**
 * Upper bound on the unsigned extension of an n bit signed interval
 * @author Edoardo Manino.
 * @brief Maximum value of unsigned extend(x, n) given n bits and signed x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param n Number of bits of the input
 */
UINT_X signed_2_unsigned_max_extend(INT_X a, INT_X b, UINT_X n)
{
  // isolate the sign bit of the input
  UINT_X sign_bit = 1 << (n - 1);

  // set the n least significant bits to one
  UINT_X mask = sign_bit | (sign_bit - 1);

  // if a <= 0 and b > 0, return the n-bit representation of -1
  if ((a & sign_bit) != 0 && (b & sign_bit) == 0)
    return -1 & mask;

  // otherwise, extend the upper bound
  return b & mask;
}

/**
 * Lower bound on the unsigned extension of an n bit signed interval
 * @author Edoardo Manino.
 * @brief Minimum value of unsigned extend(x, n) given n bits and signed x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param n Number of bits of the input
 */
UINT_X signed_2_unsigned_min_extend(INT_X a, INT_X b, UINT_X n)
{
  // isolate the sign bit of the input
  UINT_X sign_bit = 1 << (n - 1);

  // set the n least significant bits to one
  UINT_X mask = sign_bit | (sign_bit - 1);

  // if a <= 0 and b > 0, return zero
  if ((a & sign_bit) != 0 && (b & sign_bit) == 0)
    return 0;

  // otherwise, extend the upper bound
  return a & mask;
}

/**
 * Implementation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Maximum value of x | y given x in [a,b] and y in [c,d], potentially negative
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
INT_X signed_max_or(INT_X a, INT_X b, INT_X c, INT_X d)
{
  if (a < 0)
  {
    if (b < 0)
    {
      if (c < 0)
      {
        if (d < 0)
        {
          return unsigned_max_or(a, b, c, d); // a < 0, b < 0, c < 0, d < 0
        }
        else
        {
          return -1; // a < 0, b < 0, c < 0, d >= 0
        }
      }
      else
      {
        return unsigned_max_or(a, b, c, d); // a < 0, b < 0, c >= 0, d >= 0
      }
    }
    else
    {
      if (c < 0)
      {
        if (d < 0)
        {
          return -1; // a < 0, b >= 0, c < 0, d < 0
        }
        else
        {
          return unsigned_max_or(0, b, 0, d); // a < 0, b >= 0, c < 0, d >= 0
        }
      }
      else
      {
        return unsigned_max_or(0, b, c, d); // a < 0, b >= 0, c >= 0, d >= 0
      }
    }
  }
  else
  {
    if (c < 0)
    {
      if (d < 0)
      {
        return unsigned_max_or(a, b, c, d); // a >= 0, b >= 0, c < 0, d < 0
      }
      else
      {
        return unsigned_max_or(a, b, 0, d); // a >= 0, b >= 0, c < 0, d >= 0
      }
    }
    else
    {
      return unsigned_max_or(a, b, c, d); // a >= 0, b >= 0, c >= 0, d >= 0
    }
  }
}

/**
 * Implementation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Minimum value of x | y given x in [a,b] and y in [c,d], potentially negative
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
INT_X signed_min_or(INT_X a, INT_X b, INT_X c, INT_X d)
{
  if (a < 0)
  {
    if (b < 0)
    {
      if (c < 0)
      {
        if (d < 0)
        {
          return unsigned_min_or(a, b, c, d); // a < 0, b < 0, c < 0, d < 0
        }
        else
        {
          return a; // a < 0, b < 0, c < 0, d >= 0
        }
      }
      else
      {
        return unsigned_min_or(a, b, c, d); // a < 0, b < 0, c >= 0, d >= 0
      }
    }
    else
    {
      if (c < 0)
      {
        if (d < 0)
        {
          return c; // a < 0, b >= 0, c < 0, d < 0
        }
        else
        {
          return (a < c) ? a : c; // a < 0, b >= 0, c < 0, d >= 0
        }
      }
      else
      {
        return unsigned_min_or(a, -1, c, d); // a < 0, b >= 0, c >= 0, d >= 0
      }
    }
  }
  else
  {
    if (c < 0)
    {
      if (d < 0)
      {
        return unsigned_min_or(a, b, c, d); // a >= 0, b >= 0, c < 0, d < 0
      }
      else
      {
        return unsigned_min_or(a, b, c, -1); // a >= 0, b >= 0, c < 0, d >= 0
      }
    }
    else
    {
      return unsigned_min_or(a, b, c, d); // a >= 0, b >= 0, c >= 0, d >= 0
    }
  }
}

/**
 * Implementation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Maximum value of x & y given x in [a,b] and y in [c,d], potentially negative
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
INT_X signed_max_and(INT_X a, INT_X b, INT_X c, INT_X d)
{
  return ~signed_min_or(~b, ~a, ~d, ~c);
}

/**
 * Implementation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Minimum value of x & y given x in [a,b] and y in [c,d], potentially negative
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
INT_X signed_min_and(INT_X a, INT_X b, INT_X c, INT_X d)
{
  return ~signed_max_or(~b, ~a, ~d, ~c);
}

/**
 * Original upper bound on signed interval XOR, inspired by the case-based algorithm
 * for signed interval OR in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Maximum value of x ^ y given x in [a,b] and y in [c,d], potentially negative
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
INT_X signed_max_xor(INT_X a, INT_X b, INT_X c, INT_X d)
{
  if (a < 0)
  {
    if (b < 0)
    {
      if (c < 0)
      {
        if (d < 0)
        {
          return unsigned_max_xor(a, b, c, d); // a < 0, b < 0, c < 0, d < 0
        }
        else
        {
          return unsigned_max_xor(a, b, c, -1); // a < 0, b < 0, c < 0, d >= 0
        }
      }
      else
      {
        return unsigned_max_xor(a, b, c, d); // a < 0, b < 0, c >= 0, d >= 0
      }
    }
    else
    {
      if (c < 0)
      {
        if (d < 0)
        {
          return unsigned_max_xor(a, -1, c, d); // a < 0, b >= 0, c < 0, d < 0
        }
        else
        {
          INT_X e = unsigned_max_xor(a, -1, c, -1);
          INT_X f = unsigned_max_xor(0, b, 0, d);
          return (e > f) ? e : f; // a < 0, b >= 0, c < 0, d >= 0
        }
      }
      else
      {
        return unsigned_max_xor(0, b, c, d); // a < 0, b >= 0, c >= 0, d >= 0
      }
    }
  }
  else
  {
    if (c < 0)
    {
      if (d < 0)
      {
        return unsigned_max_xor(a, b, c, d); // a >= 0, b >= 0, c < 0, d < 0
      }
      else
      {
        return unsigned_max_xor(a, b, 0, d); // a >= 0, b >= 0, c < 0, d >= 0
      }
    }
    else
    {
      return unsigned_max_xor(a, b, c, d); // a >= 0, b >= 0, c >= 0, d >= 0
    }
  }
}

/**
 * Original lower bound on signed interval XOR, inspired by the case-based algorithm
 * for signed interval OR in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Maximum value of x ^ y given x in [a,b] and y in [c,d], potentially negative
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
INT_X signed_min_xor(INT_X a, INT_X b, INT_X c, INT_X d)
{
  if (a < 0)
  {
    if (b < 0)
    {
      if (c < 0)
      {
        if (d < 0)
        {
          return unsigned_min_xor(a, b, c, d); // a < 0, b < 0, c < 0, d < 0
        }
        else
        {
          return unsigned_min_xor(a, b, 0, d); // a < 0, b < 0, c < 0, d >= 0
        }
      }
      else
      {
        return unsigned_min_xor(a, b, c, d); // a < 0, b < 0, c >= 0, d >= 0
      }
    }
    else
    {
      if (c < 0)
      {
        if (d < 0)
        {
          return unsigned_min_xor(0, b, c, d); // a < 0, b >= 0, c < 0, d < 0
        }
        else
        {
          INT_X e = unsigned_min_xor(a, -1, 0, d);
          INT_X f = unsigned_min_xor(0, b, c, -1);
          return (e < f) ? e : f; // a < 0, b >= 0, c < 0, d >= 0
        }
      }
      else
      {
        return unsigned_min_xor(a, -1, c, d); // a < 0, b >= 0, c >= 0, d >= 0
      }
    }
  }
  else
  {
    if (c < 0)
    {
      if (d < 0)
      {
        return unsigned_min_xor(a, b, c, d); // a >= 0, b >= 0, c < 0, d < 0
      }
      else
      {
        return unsigned_min_xor(a, b, c, -1); // a >= 0, b >= 0, c < 0, d >= 0
      }
    }
    else
    {
      return unsigned_min_xor(a, b, c, d); // a >= 0, b >= 0, c >= 0, d >= 0
    }
  }
}

/**
 * Implementation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Maximum value of ~x given x in [a,b], potentially negative
 * @param a Min value of variable x
 * @param b Max value of variable x
 */
INT_X signed_max_not(INT_X a, INT_X)
{
  return ~a;
}

/**
 * Implementation of the algorithm in Henry S. Warren Jr., "Hacker's Delight", 2003
 * @author Edoardo Manino.
 * @brief Minimum value of ~x given x in [a,b], potentially negative
 * @param a Min value of variable x
 * @param b Max value of variable x
 */
INT_X signed_min_not(INT_X, INT_X b)
{
  return ~b;
}

/**
 * Upper bound on interval left shift
 * @author Edoardo Manino.
 * @brief Maximum value of x << y given x in [a,b] and y in [c,d], potentially negative
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
INT_X signed_max_left_shift(INT_X, INT_X b, UINT_X c, UINT_X d)
{
  if (b >= 0)
    return b << d;
  else
    return b << c;
}

/**
 * Lower bound on interval left shift
 * @author Edoardo Manino.
 * @brief Minimum value of x << y given x in [a,b] and y in [c,d], potentially negative
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
INT_X signed_min_left_shift(INT_X a, INT_X, UINT_X c, UINT_X d)
{
  if (a >= 0)
    return a << c;
  else
    return a << d;
}

/**
 * Upper bound on interval right shift
 * @author Edoardo Manino.
 * @brief Maximum value of x >> y given x in [a,b] and y in [c,d], potentially negative
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
INT_X signed_max_right_shift(INT_X, INT_X b, UINT_X c, UINT_X d)
{
  if (b >= 0)
    return b >> c;
  else
    return b >> d;
}

/**
 * Lower bound on interval right shift
 * @author Edoardo Manino.
 * @brief Minimum value of x >> y given x in [a,b] and y in [c,d], potentially negative
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param c Min value of variable y
 * @param d Max value of variable y
 */
INT_X signed_min_right_shift(INT_X a, INT_X, UINT_X c, UINT_X d)
{
  if (a >= 0)
    return a >> d;
  else
    return a >> c;
}

/**
 * Cast a limited-width unsigned integer to signed integer
 * @author Edoardo Manino.
 * @brief Compute cas(x, n) given an n-bit unsigned input x
 * @param x Value to be converted
 * @param n Number of bits of the result
 */
INT_X unsigned_2_signed(UINT_X x, UINT_X n)
{
  // isolate the sign bit of the truncated output
  UINT_X sign_bit = 1 << (n - 1);

  // set the n least significant bits to one
  UINT_X mask = sign_bit | (sign_bit - 1);

  // if the input is positive, just truncate and cast
  if ((x & sign_bit) == 0)
    return (INT_X)x & mask;

  // if the input is negative, convert it to positive with relation -x = (~x) + 1,
  // truncate it, cast it to signed integer and return the negative result
  INT_X positive_x = (~x & mask) + 1;
  return -positive_x;
}

/**
 * Upper bound on unsigned to signed interval truncation
 * @author Edoardo Manino.
 * @brief Maximum value of signed truncate(x, n) given n bits and unsigned x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param n Number of bits of the result
 */
INT_X unsigned_2_signed_max_truncate(UINT_X a, UINT_X b, UINT_X n)
{
  // isolate the sign bit of the truncated output
  UINT_X sign_bit = 1 << (n - 1);

  // keep the bits of the upper bound that are at least as significant
  // as the sign bit of the truncated output
  // e.g. b = 0b01101010 and n = 4 yields cand_bits = 0b01101000
  UINT_X cand_bits = b & (-sign_bit);

  // isolate the least significant candidate bit
  // e.g. cand_bits = 0b01101010 yields least_bit = 0b00000010
  UINT_X least_bit = cand_bits & (-cand_bits);

  // reduce b by removing least_bit and activating all the less significant ones
  // e.g. b = 0b00101100 and least_bit = 0b00001000 yields alternate_b = 0b00100111
  UINT_X alternate_b = (b - least_bit) | (least_bit - 1);

  // reduce b further by setting the sign bit to zero
  // this operation ensures that the result is positive
  UINT_X positive_b = alternate_b & ~sign_bit;

  // check if the reduced positive b still belongs to the input interval
  if (positive_b >= a && positive_b <= b)
    return unsigned_2_signed(positive_b, n);

  // otherwise, try to use the upper bound (if positive)
  if ((b & sign_bit) == 0)
    return unsigned_2_signed(b, n);

  // if we are forced to return a negative number, try to maximise it
  if (alternate_b >= a && alternate_b <= b)
    return unsigned_2_signed(alternate_b, n);

  // otherwise, just use the negative upper bound
  return unsigned_2_signed(b, n);
}

/**
 * Lower bound on unsigned to signed interval truncation
 * @author Edoardo Manino.
 * @brief Minimum value of signed truncate(x, n) given n bits and unsigned x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param n Number of bits of the result
 */
INT_X unsigned_2_signed_min_truncate(UINT_X a, UINT_X b, UINT_X n)
{
  // isolate the sign bit of the truncated output
  UINT_X sign_bit = 1 << (n - 1);

  // flip the bits of the upper bound that are at least as significant
  // as the sign bit of the truncated output
  // e.g. a = 0b01100010 and n = 4 yields cand_bits = 0b10011000
  UINT_X cand_bits = ~a & (-sign_bit);

  // isolate the least significant candidate bit
  // e.g. cand_bits = 0b01101010 yields least_bit = 0b00000010
  UINT_X least_bit = cand_bits & (-cand_bits);

  // increase a by adding least_bit and removing all the less significant ones
  // e.g. a = 0b00110100 and least_bit = 0b00001000 yields alternate_a = 0b00111000
  UINT_X alternate_a = (a | least_bit) & (-least_bit);

  // increase a further by setting the sign bit to one
  // this operation ensures that the result is positive
  UINT_X negative_a = alternate_a | sign_bit;

  // check if the increased negative a still belongs to the input interval
  if (negative_a >= a && negative_a <= b)
    return unsigned_2_signed(negative_a, n);

  // otherwise, try to use the lower bound (if negative)
  if ((a & sign_bit) != 0)
    return unsigned_2_signed(a, n);

  // if we are forced to return a positive number, try to minimise it
  if (alternate_a >= a && alternate_a <= b)
    return unsigned_2_signed(alternate_a, n);

  // otherwise, just use the upper bound
  return unsigned_2_signed(a, n);
}

/**
 * Upper bound on signed to signed interval truncation
 * @author Edoardo Manino.
 * @brief Maximum value of signed truncate(x, n) given n bits and signed x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param n Number of bits of the result
 */
INT_X signed_2_signed_max_truncate(INT_X a, INT_X b, UINT_X n)
{
  // if the input interval contains zero, split it
  if (a < 0 && b >= 0)
  {
    INT_X c = unsigned_2_signed_max_truncate(a, -1, n);
    INT_X d = unsigned_2_signed_max_truncate(0, b, n);
    return (c > d) ? c : d;
  }

  // otherwise interpret [a,b] as an unsigned interval
  return unsigned_2_signed_max_truncate(a, b, n);
}

/**
 * Lower bound on signed to signed interval truncation
 * @author Edoardo Manino.
 * @brief Minimum value of signed truncate(x, n) given n bits and signed x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param n Number of bits of the result
 */
INT_X signed_2_signed_min_truncate(INT_X a, INT_X b, UINT_X n)
{
  // if the input interval contains zero, split it
  if (a < 0 && b >= 0)
  {
    INT_X c = unsigned_2_signed_min_truncate(a, -1, n);
    INT_X d = unsigned_2_signed_min_truncate(0, b, n);
    return (c < d) ? c : d;
  }

  // otherwise interpret [a,b] as an unsigned interval
  return unsigned_2_signed_min_truncate(a, b, n);
}

/**
 * Upper bound on the signed extension of an n bit unsigned interval
 * @author Edoardo Manino.
 * @brief Maximum value of signed extend(x, n) given n bits and unsigned x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param n Number of bits of the input
 */
INT_X unsigned_2_signed_max_extend(UINT_X a, UINT_X b, UINT_X n)
{
  // isolate the sign bit of the input
  UINT_X sign_bit = 1 << (n - 1);

  // if a > 0 and b <= 0, return the largest positive number
  if ((b & sign_bit) != 0)
  {
    if ((a & sign_bit) == 0)
      return sign_bit - 1;

    // if both a <= 0 and b <= 0, extend the sign of the upper bound
    return unsigned_2_signed(b, n);
  }

  // otherwise, keep the positive upper bound
  return b;
}

/**
 * Lower bound on the signed extension of an n bit unsigned interval
 * @author Edoardo Manino.
 * @brief Minimum value of signed extend(x, n) given n bits and unsigned x in [a,b]
 * @param a Min value of variable x
 * @param b Max value of variable x
 * @param n Number of bits of the input
 */
INT_X unsigned_2_signed_min_extend(UINT_X a, UINT_X b, UINT_X n)
{
  // isolate the sign bit of the input
  UINT_X sign_bit = 1 << (n - 1);

  // if a > 0 and b <= 0, return the smallest negative number
  if ((b & sign_bit) != 0)
  {
    if ((a & sign_bit) == 0)
      return unsigned_2_signed(sign_bit, n);

    // if both a <= 0 and b <= 0, extend the sign of the lower bound
    return unsigned_2_signed(a, n);
  }

  // otherwise, keep the positive lower bound
  return a;
}

#define UINT_FUNC(FUNC, LHS, RHS)                                              \
  FUNC(                                                                        \
    LHS.get_lower().to_uint64(),                                               \
    LHS.get_upper().to_uint64(),                                               \
    RHS.get_lower().to_uint64(),                                               \
    RHS.get_upper().to_uint64())

#define INT_FUNC(FUNC, LHS, RHS)                                               \
  FUNC(                                                                        \
    LHS.get_lower().to_int64(),                                                \
    LHS.get_upper().to_int64(),                                                \
    RHS.get_lower().to_int64(),                                                \
    RHS.get_upper().to_int64())

#define GET_BIT_INTERVALS(NAME, MIN_UFUNC, MAX_UFUNC, MIN_FUNC, MAX_FUNC)      \
  template <>                                                                  \
  interval_templatet<BigInt> interval_templatet<BigInt>::NAME(                 \
    const interval_templatet<BigInt> &lhs,                                     \
    const interval_templatet<BigInt> &rhs) const                               \
  {                                                                            \
    interval_templatet<BigInt> result;                                         \
    if (!lhs.lower || !lhs.upper || !rhs.lower || !rhs.upper)                  \
      return result;                                                           \
    if (is_unsignedbv_type(lhs.type) && is_unsignedbv_type(rhs.type))          \
    {                                                                          \
      result.set_lower(UINT_FUNC(MIN_UFUNC, lhs, rhs));                        \
      result.set_upper(UINT_FUNC(MAX_UFUNC, lhs, rhs));                        \
    }                                                                          \
    else if (is_signedbv_type(lhs.type) && is_signedbv_type(rhs.type))         \
    {                                                                          \
      result.set_lower(INT_FUNC(MIN_FUNC, lhs, rhs));                          \
      result.set_upper(INT_FUNC(MAX_FUNC, lhs, rhs));                          \
    }                                                                          \
    return result;                                                             \
  }

#include <big-int/bigint.hh>

GET_BIT_INTERVALS(
  interval_bitand,
  unsigned_min_and,
  unsigned_max_and,
  signed_min_and,
  signed_max_and)

GET_BIT_INTERVALS(
  interval_bitor,
  unsigned_min_or,
  unsigned_max_or,
  signed_min_or,
  signed_max_or)

GET_BIT_INTERVALS(
  interval_bitxor,
  unsigned_min_xor,
  unsigned_max_xor,
  signed_min_xor,
  signed_max_xor)

GET_BIT_INTERVALS(
  interval_logical_right_shift,
  unsigned_min_right_shift,
  unsigned_max_right_shift,
  signed_min_right_shift,
  signed_max_right_shift)

GET_BIT_INTERVALS(
  interval_left_shift,
  unsigned_min_left_shift,
  unsigned_max_left_shift,
  signed_min_left_shift,
  signed_max_left_shift)

#undef UINT_FUNC
#undef INT_FUNC
#undef GET_BIT_INTERVALS

#endif // BITWISE_BOUNDS_H_INCLUDED
