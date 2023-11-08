
import math
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

def compound_interest(n):
    """
    compound_interest calculates future value using the classical compound interest formula for a fixed period of 1 year:
    A = P(1+r/n)^(nt)
    
    A: The future value of the investment/loan, including interest.
    P: The principal amount (the initial sum of money), which is assumed to be 1 in this function.
    r: The annual interest rate (as a proportion, e.g., 1 for 100%), which is assumed to be 1 in this function.
    n: The number of times that interest is compounded per year.
    t: The time the money is invested or borrowed for, which is fixed at 1 year in this function.

    :param n: compounding frequency per year
    :return: The amount of money accumulated after 1 year, including interest
    """ 
    
    # The annual interest rate
    annual_interest_rate = 1
    
    # The initial sum of money  
    principal_amount = 1
    
    A = principal_amount*(1+(annual_interest_rate/n))**(n*1)
    
    return A


# Display the growth of the investment for different compounding frequencies
print(f"Investment grow when interest is compounded annually {compound_interest(1)}")
print(f"Investment grow when interest is compounded twice per year {compound_interest(2)}")
print(f"Investment grow when interest is compounded quarterly {compound_interest(4)}")
print(f"Investment grow when interest is compounded monthly {compound_interest(12)}")
print(f"Investment grow when interest is compounded daily {compound_interest(365)}")
print(f"Investment grow when interest is compounded two times per day {compound_interest(365*2)}")
print(f"Investment grow when interest is compounded 24 times per day {compound_interest(365*24)}")
print(f"Investment grow when interest is compounded 192 times per day {compound_interest(365*192)}")

# Euler's number (e) for comparison to demonstrate the limit as n approaches infinity
print(f"Euler's number {math.exp(1)}") 



# Create lists to store the values of n and the corresponding compound interest
n_values = range(1, 1001)
interest_values = [compound_interest(n) for n in n_values]

# Plot the compound interest values
plt.figure(figsize=(10, 5))  # Set the figure size
plt.plot(n_values, interest_values, label='Compound Interest', color='blue')

# Add a horizontal line for Euler's number
plt.axhline(y=math.exp(1), color='red', linestyle='--', label=f"Euler's Number (e ≈ {math.exp(1):.2f})")

# Add labels and title
plt.xlabel('n (Number of times interest is compounded per year)')
plt.ylabel('Future Value after 1 year')
plt.title('Compound Interest and Euler\'s Number')
plt.legend()  # Show the legend

# Improve grid and layout
plt.grid(True)
plt.tight_layout()
plt.show()

a = [-8, -99, 10, 0, 50, 6089, 4879, -5, 0, 66, -900, 10]
a_sum = 0
for ai in a:
    a_sum += ai
    
print(a_sum)