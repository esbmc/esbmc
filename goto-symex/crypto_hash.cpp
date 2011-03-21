#include "crypto_hash.h"

extern "C" {
  #include <openssl/sha.h>
  #include <string.h>
}

bool
crypto_hash::operator<(const crypto_hash h2) const
{

  if (memcmp(hash, h2.hash, 32) < 0)
    return true;

  return false;
}

std::string
crypto_hash::to_string() const
{
  int i;
  char hex[65];

  for (i = 0; i < 32; i++)
    sprintf(&hex[i*2], "%02X", (unsigned char)hash[i]);

  hex[64] = '\0';
  return std::string(hex);
}

void
crypto_hash::init(const uint8_t *data, int sz)
{
  SHA256_CTX c;

  SHA256_Init(&c);
  SHA256_Update(&c, data, sz);
  SHA256_Final(hash, &c);
  valid = true;
  return;
}

crypto_hash::crypto_hash(const uint8_t *data, int sz)
{

  init(data, sz);
}

crypto_hash::crypto_hash(std::string str)
{

  init((const uint8_t *)str.data(), str.length());
}

crypto_hash::crypto_hash()
{

  valid = false;
}
