/* Solidity ether and time unit conversion functions */
#include "solidity_types.h"

// ether
uint256_t _ESBMC_wei(uint256_t x)
{
__ESBMC_HIDE:;
  return x;
}
uint256_t _ESBMC_gwei(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)1000000000; // 10^9
}
uint256_t _ESBMC_szabo(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)1000000000000; // 10^12
}
uint256_t _ESBMC_finney(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)1000000000000000; // 10^15
}
uint256_t _ESBMC_ether(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)1000000000000000000; // 10^18
}

// time
uint256_t _ESBMC_seconds(uint256_t x)
{
__ESBMC_HIDE:;
  return x;
}
uint256_t _ESBMC_minutes(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)60;
}
uint256_t _ESBMC_hours(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)3600; // 60 * 60
}
uint256_t _ESBMC_days(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)86400; // 24 * 3600
}
uint256_t _ESBMC_weeks(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)604800; // 7 * 86400
}
uint256_t _ESBMC_years(uint256_t x)
{
__ESBMC_HIDE:;
  return x * (uint256_t)31536000; // 365 * 86400
}
