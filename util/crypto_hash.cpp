
#include "crypto_hash.h"

#ifndef NO_OPENSSL

extern "C" {
  #include <dlfcn.h>
  #include <string.h>

  #include <openssl/sha.h>
}

bool crypto_hash::have_pointers = false;
int (*crypto_hash::sha_init)(SHA256_CTX *c) = 0;
int (*crypto_hash::sha_update)(SHA256_CTX *c, const void *data, size_t len) = 0;
int (*crypto_hash::sha_final)(unsigned char *md, SHA256_CTX *c) = 0;

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

crypto_hash::crypto_hash()
{

  if (!have_pointers)
    setup_pointers();

  sha_init(&c);
}

void
crypto_hash::ingest(void const *data, unsigned int size)
{

  sha_update(&c, data, size);
  return;
}

void
crypto_hash::fin(void)
{

  sha_final(hash, &c);
}

void
crypto_hash::setup_pointers()
{
  void *ssl_lib;
  long (*ssleay)(void);

  ssl_lib = dlopen("libcrypto.so", RTLD_LAZY);
  if (ssl_lib == NULL)
    throw "Couldn't open OpenSSL crypto library - can't hash state";

  ssleay = (long (*)(void)) dlsym(ssl_lib, "SSLeay");
  // Check for version 0.9.8 - I believe this is the first release with SHA256.
  if (ssleay() < 0x000908000)
    throw "OpenSSL >= 0.9.8 required for state hashing";

  sha_init = (int (*) (SHA256_CTX *c)) dlsym(ssl_lib, "SHA256_Init");
  sha_update = (int (*) (SHA256_CTX *c, const void *data, size_t len))
               dlsym(ssl_lib, "SHA256_Update");
  sha_final = (int (*) (unsigned char *md, SHA256_CTX *c))
               dlsym(ssl_lib, "SHA256_Final");
  have_pointers = true;
  return;
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
  return false;
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
