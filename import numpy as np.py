import numpy as np
import matplotlib.pyplot as plt

#function to compute f(x)
def f(x):
    return np.exp(x**2)

#function for user input to determine the number of nodes
def user_input(default_exponent=4):
    exponent_input = input("Enter the degree of the interpolation polynomial (default is 4): ")
    try:
        exponent = int(exponent_input) if exponent_input else default_exponent
    except ValueError:
        print("Invalid input! Using default degree.")
        exponent = default_exponent
    return exponent + 1

#function to solve a tridiagonal system using the прогонки method
def tridiagonal_solve(a, b, c, d):
    n = len(d)
    
    #forward elimination
    c_prime = [0] * (n - 1)
    d_prime = [0] * n
    
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n):
        denominator = b[i] - a[i - 1] * c_prime[i - 1]
        c_prime[i - 1] = c[i] / denominator if i < n - 1 else 0
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denominator
    
    #back substitution
    x = [0] * n
    x[-1] = d_prime[-1]
    
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    
    return x

#function to compute the cubic spline interpolation
def spline(x_val, x, coefficients, n):
    for i in range(n):
        if x[i] <= x_val <= x[i+1]:
            m_i, m_ip1, Ai, Bi, h_i = coefficients[i]
            xi, xip1 = x[i], x[i+1]
            return (
                m_i * (xip1 - x_val)**3 / (6 * h_i)
                + m_ip1 * (x_val - xi)**3 / (6 * h_i)
                + Ai * (xip1 - x_val) / h_i
                + Bi * (x_val - xi) / h_i
            )
    return None

#main program block
if __name__ == "__main__":
    #input data
    a, b = -1, 4
    n = user_input()

    #interval subdivision
    x = np.linspace(a, b, n+1)
    f_values = f(x)
    h = np.diff(x)

    #matrix and right-hand side of the system
    a_diag = [h[i-1] / 6 for i in range(1, n)]
    b_diag = [(h[i-1] + h[i]) / 3 for i in range(1, n)]
    c_diag = [h[i] / 6 for i in range(1, n)]
    rhs = [(f_values[i+1] - f_values[i]) / h[i] - (f_values[i] - f_values[i-1]) / h[i-1] for i in range(1, n)]

    #solving the system using the tridiagonal solver
    m = [0] * (n+1)  # m0 = 0, mn = 0
    m[1:n] = tridiagonal_solve(a_diag, b_diag, c_diag, rhs)

    #storing coefficients
    coefficients = []
    for i in range(n):
        Ai = f_values[i] - m[i] * h[i]**2 / 6
        Bi = f_values[i+1] - m[i+1] * h[i]**2 / 6
        coefficients.append((m[i], m[i+1], Ai, Bi, h[i]))

    #test computation
    x_test = np.linspace(a, b, 100)
    spline_values = [spline(xi, x, coefficients, n) for xi in x_test]

    #visualization
    plt.plot(x_test, spline_values, label='Spline')
    plt.plot(x, f_values, 'o', label='Nodes')
    plt.plot(x_test, f(x_test), '--', label='Original Function')
    plt.legend()
    plt.show()
