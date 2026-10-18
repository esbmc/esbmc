/*
 * Full model matching Solidity GOTO as closely as possible.
 * Includes C's constructor calling D(4) which also calls transfer.
 */
#include <assert.h>

typedef unsigned _ExtInt(256) uint256_t;
typedef unsigned _ExtInt(160) addr_t;

uint256_t nondet_uint256(void);
addr_t nondet_addr(void);
_Bool nondet_bool(void);

struct Contract_D {
    addr_t    address;
    uint256_t balance;
    uint256_t x;
};

struct Contract_C {
    struct Contract_D *d;
    addr_t    address;
    uint256_t balance;
};

struct Contract_C C_instance;
struct Contract_D D_instance;

/* Global msg context (like Solidity) */
addr_t msg_sender;
uint256_t msg_value;

void transfer_D(struct Contract_D *src, addr_t dest_addr, uint256_t val)
{
    if (dest_addr == D_instance.address)
    {
        if (src->balance < val)
            __ESBMC_assume(0);
        src->balance -= val;
        D_instance.balance += val;
        return;
    }
    if (dest_addr == C_instance.address)
    {
        if (src->balance < val)
            __ESBMC_assume(0);
        src->balance -= val;
        C_instance.balance += val;
        return;
    }
}

void D_constructor(struct Contract_D *d, uint256_t a)
{
    d->address = nondet_addr();
    d->balance = msg_value;  /* payable constructor init */
    d->x = (uint256_t)0;
    /* payable(msg.sender).transfer(1 ether) */
    transfer_D(d, msg_sender, (uint256_t)1000000000000000000ULL);
    d->x = a;
}

void C_constructor(struct Contract_C *c)
{
    c->address = nondet_addr();
    c->balance = nondet_uint256();
    /* D d = new D(4) — creates D inside C's constructor */
    struct Contract_D tmp_d;
    D_constructor(&tmp_d, (uint256_t)4);
    /* (in real GOTO: c->d = &heap_copy, but we skip that) */
}

void createAndEndowD(struct Contract_C *this_ptr, uint256_t amount)
{
    uint256_t balancebefore = this_ptr->balance;

    /* model_transaction: save/set msg context */
    addr_t old_sender = msg_sender;
    uint256_t old_value = msg_value;
    msg_sender = this_ptr->address;
    msg_value = amount;

    if (this_ptr->balance < amount)
        goto end;
    this_ptr->balance = this_ptr->balance - amount;

    {
        struct Contract_D tmp_D;
        D_constructor(&tmp_D, amount);

        /* Restore msg context */
        msg_sender = old_sender;
        msg_value = old_value;

        uint256_t balanceafter = this_ptr->balance;
        assert(balanceafter == balancebefore - amount);
    }
    return;
end:
    msg_sender = old_sender;
    msg_value = old_value;
}

void nondet_extcall(void)
{
    if (nondet_bool())
        createAndEndowD(&C_instance, nondet_uint256());
}

void main_loop(void)
{
    while (nondet_bool())
        nondet_extcall();
}

int main()
{
    /* Initialize msg context */
    msg_sender = nondet_addr();
    msg_value = nondet_uint256();
    D_instance.address = nondet_addr();
    D_instance.balance = (uint256_t)0;

    /* C constructor (includes D(4) creation) */
    struct Contract_C tmp_c;
    C_constructor(&tmp_c);
    C_instance = tmp_c;

    main_loop();

    return 0;
}
