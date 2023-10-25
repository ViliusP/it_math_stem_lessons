# Read more at: http://cissandbox.bentley.edu/sandbox/wp-content/uploads/2022-02-10-Documentation-on-f-strings-Updated.pdf

## ------------------------------------
## Basic 
## ------------------------------------
name = "Alice"
greeting = f"Hello, {name}!"

print(greeting)  # Outputs: Hello, Alice!


## ------------------------------------
## Multiple variables
## ------------------------------------
day = "Saturday"
month = "October"
date = f"Today is {day} in the month of {month}."

print(date)  # Outputs: Today is Saturday in the month of October.


## ------------------------------------
## Expression inside f-strings
## ------------------------------------
a = 5
b = 10
result = f"The sum of {a} and {b} is {a+b}."

print(result)


## ------------------------------------
## In function
## ------------------------------------
def greet(name):
    return f"Hello, {name}!"

name = "Alice"
message = f"{greet(name)}"

print(message)


## ------------------------------------
## Formatting Numbers
## ------------------------------------
pi = 3.141592653589793
formatted = f"Pi rounded to 3 decimal places: {pi:.3f}"

print(formatted)


## ------------------------------------
## Using Flags for Formatting
## ------------------------------------
value = 12.345
formatted = f"Right aligned with width 10: {value:>10.2f}"

print(formatted)


## ------------------------------------
## Date formatting
## ------------------------------------
from datetime import datetime
now = datetime.now()

formatted_date = f"Today's date and time: {now:%Y-%m-%d %H:%M:%S}"

print(formatted_date)


## ------------------------------------
## Conditional expressions
## ------------------------------------
age = 20
status = f"You are {'an adult' if age >= 18 else 'a minor'}."

print(status)


## ------------------------------------
## Loops
## ------------------------------------
numbers = [1, 2, 3]
output = "\n".join(f"Number: {num}" for num in numbers)

print(output)
