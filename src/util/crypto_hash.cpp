
#include "crypto_hash.h"
#include <ac_config.h>

#ifndef HAVE_OPENSSL

extern "C" {
#include <openssl/sha.h>
}

#include <cstring>

class crypto_hash_private {
public:
#ifndef HAVE_OPENSSL
  SHA_CTX c;
#endif
};

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

crypto_hash::crypto_hash() : p_crypto(std::make_shared<crypto_hash_private>())
{
  SHA1_Init(&p_crypto->c);
}

void
crypto_hash::ingest(void const *data, unsigned int size)
{
  SHA1_Update(&p_crypto->c, data, size);
  return;
}

void
crypto_hash::fin(void)
{
  SHA1_Final(hash, &p_crypto->c);
}

#else /* !NO_OPENSSL */

extern "C" {
  #include <stdlib.h>
  #include <string.h>
};

#include <iostream>

/* Generate some dummy implementations that complain and abort if used */

bool
crypto_hash::operator<(const crypto_hash h2 __attribute__((unused))) const
{

  abort();
  return false;
}

std::string
crypto_hash::to_string() const
{

  abort();
}

crypto_hash::crypto_hash()
{
  // Valid; some exist as default constructions within other parts of ESBMC.
  // Preventing this constructor running leads to *all* runtimes being blocked
  // by errors thrown from here.
}

void
crypto_hash::ingest(const void *data __attribute__((unused)),
                    unsigned int size __attribute__((unused)))
{
  abort();
}

void
crypto_hash::fin(void)
{
  abort();
}



#endif /* NO_OPENSSL */
