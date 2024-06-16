from typing import List, Tuple

# For storing Vaida's and User's passwords
class Password: 
    def __init__(self, value: str, lenght: int, uppercase_count: int, lowercase_count: int, digits_count: int, special_symbols_count: int, strength: str):
        self.value = value
        self.lenght = lenght
        self.uppercase_count = uppercase_count
        self.lowercase_count = lowercase_count
        self.digits_count = digits_count
        self.special_symbols_count = special_symbols_count
        self.strength = strength

    # Parses password from raw string 
    def from_line(raw_line: str) -> 'Password':
        parts = raw_line.split()
        length = int(parts[1])
        uppercase_count = int(parts[2])
        lowercase_count = int(parts[3])
        digits_count = int(parts[4])
        special_symbols_count = int(parts[5])
        strength = None
        if len(parts) == 7:
            strength = parts[6]

        return Password(parts[0], length, uppercase_count, lowercase_count, digits_count, special_symbols_count, strength)
    
    # Finds similar passwords of given list
    def similar_to(self, passwords: List['Password']) -> Tuple[List['Password'], int]:
        similar_passwords = []
        min_similiarity = 0
        for password in passwords:
            similiarity = self.calc_similiarity(password)
            if(min_similiarity == 0):
                min_similiarity = similiarity

            if(similiarity < min_similiarity):
                similar_passwords = [password]
                min_similiarity = similiarity
            elif(similiarity == min_similiarity):
                similar_passwords.append(password)

        return similar_passwords, min_similiarity

    # Calculate similiarity value by summing all absolute differences of passwords strength values 
    def calc_similiarity(self, password: 'Password'):
        similiarity = 0
        similiarity += abs(self.lenght - password.lenght) 
        similiarity += abs(self.uppercase_count - password.uppercase_count) 
        similiarity += abs(self.lowercase_count - password.lowercase_count) 
        similiarity += abs(self.digits_count - password.digits_count) 
        similiarity += abs(self.special_symbols_count - password.special_symbols_count) 
        return similiarity

# Program starts here
file=open("U2.txt")
content = file.read().splitlines()
file.close()

result_file=open("U2rez.txt", "w")

password_counts = content[0].split()

user_passwords_count = int(password_counts[0])

user_passwords: List[Password] = []
vaida_passwords: List[Password] = []

for raw_user_password in content[1:user_passwords_count+1]:
    user_password = Password.from_line(raw_user_password)
    user_passwords.append(user_password)

for raw_vaida_password in content[user_passwords_count+1:]:
    vaida_password = Password.from_line(raw_vaida_password)
    vaida_passwords.append(vaida_password)

for user_password in user_passwords:
    similar_passwords, similiarity = user_password.similar_to(vaida_passwords)
    similar_passwords.sort(key= lambda x: x.lenght, reverse=True)
    result_file.write(f"{user_password.value:<14} {similar_passwords[0].strength} {similiarity}\n")
    for similar_password in similar_passwords:
        result_file.write(f"{similar_password.value}\n")

result_file.close()