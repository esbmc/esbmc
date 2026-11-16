struct tickets
{
  unsigned short head, tail;
};
union ticketpair
{
  unsigned int head_tail;
  struct tickets t;
};
struct raw_spinlock
{
  union ticketpair raw_lock;
  unsigned int magic, owner_cpu;
  void *owner;
};
union spinlock
{
  struct raw_spinlock rlock;
};
struct icount
{
  unsigned int cts, dsr, rng, dcd, rx, tx, frame, overrun, parity, brk;
};
struct uart_port
{
  union spinlock lock;
  unsigned long iobase;
  unsigned char *membase;
  void (*f0)(void), (*f1)(void), (*f2)(void), (*f3)(void);
  unsigned int irq, uartclk, fifosize;
  unsigned char x_char, regshift, iotype, unused1;
  struct icount icount;
  unsigned int flags, mctrl, timeout, type;
  unsigned long mapbase;
  unsigned char unused[2];
  void *private_data;
};
struct jtag
{
  struct uart_port port;
  unsigned int sigs;
  unsigned long imr;
};
static struct jtag ports[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  unsigned long port = (unsigned long)ports + (unsigned long)i * sizeof(struct jtag);
  *(unsigned long *)(port + __builtin_offsetof(struct uart_port, mapbase)) = 1;
  *(unsigned int *)(port + __builtin_offsetof(struct uart_port, icount.tx)) = 2;
  *(unsigned int *)(port + __builtin_offsetof(struct uart_port, lock)) = 3;
  __ESBMC_assert(i < 2, "index in range");
  return 0;
}
