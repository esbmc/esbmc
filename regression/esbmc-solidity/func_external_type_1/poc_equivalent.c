/*
 * C equivalent of func_external_type_1 (Solidity Oracle callback example).
 * Tests: function pointers stored in structs, passed as arguments, invoked later.
 *
 * Solidity: function(uint) external callback  =>  C: void (*callback)(unsigned)
 *
 * If ESBMC backend handles function pointers in structs correctly, this should
 * produce VERIFICATION SUCCESSFUL.
 */
#include <assert.h>
#include <string.h>

/* --- Oracle contract equivalent --- */

typedef void (*callback_t)(unsigned);

struct Request {
    char data[32];
    callback_t callback;
};

#define MAX_REQUESTS 8
struct Request requests[MAX_REQUESTS];
unsigned request_count = 0;

void query(const char *data, callback_t callback)
{
    struct Request *r = &requests[request_count++];
    strncpy(r->data, data, sizeof(r->data) - 1);
    r->callback = callback;
}

void reply(unsigned requestID, unsigned response)
{
    requests[requestID].callback(response);
}

/* --- OracleUser contract equivalent --- */

unsigned exchangeRate = 0;

void oracleResponse(unsigned response)
{
    exchangeRate = response;
}

void buySomething(void)
{
    query("USD", oracleResponse);
}

int main()
{
    buySomething();

    /* Oracle replies with exchange rate 42 */
    reply(0, 42);

    /* Verify the callback was invoked and stored the value */
    assert(exchangeRate == 42);

    return 0;
}
