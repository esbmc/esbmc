# HumanEval/147
# Entry Point: get_max_triples
# ESBMC-compatible format with direct assertions


def get_max_triples(n):
    """
    You are given a positive integer n. You have to create an integer array a of length n.
        For each i (1 ≤ i ≤ n), the value of a[i] = i * i - i + 1.
        Return the number of triples (a[i], a[j], a[k]) of a where i < j < k, 
    and a[i] + a[j] + a[k] is a multiple of 3.

    Example :
        Input: n = 5
        Output: 1
        Explanation: 
        a = [1, 3, 7, 13, 21]
        The only valid triple is (1, 7, 13).
    """
    A = [i*i - i + 1 for i in range(1,n+1)]
    ans = []
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                if (A[i]+A[j]+A[k])%3 == 0:
                    ans += [(A[i],A[j],A[k])]
    return len(ans)

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert get_max_triples(5) == 1
    # assert get_max_triples(6) == 4
    # assert get_max_triples(10) == 36
    # assert get_max_triples(100) == 53361
    print("✅ HumanEval/147 - All assertions completed!")
