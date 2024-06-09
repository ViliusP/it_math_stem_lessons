def calculate_category_steps(category, students_steps): 
    result = [int(category)]
    total_travelled = 0
    ok_students = 0

    for steps in students_steps:    
        step_length = int(steps[0])
        
        steps_without_length = steps[1:]
        step_sum = 0
        zero_exists = False

        for raw_step_count in steps_without_length:
            step_count = int(raw_step_count)

            step_sum += step_count
        
            zero_exists = (step_count == 0) or zero_exists

        if(not zero_exists):
            ok_students += 1 
            total_travelled += (step_sum * step_length)
        
    result.append(ok_students)
    result.append(total_travelled)

    return result


def write_results(results):
    filename = "U1rez.txt"
    file = open(filename, "w")
    
    for result in results:
        if(result[1] != 0):
            file.write(f"{result[0]} {result[1]} {result[2]/100000:.2f}\n")
        
    file.close()
    return


def parse_categories(content): 
    categories = {}

    # line = category, step length, step per day 
    for line in content: 
        splitted_line = line.split(" ")

        category = int(splitted_line[0])
        categories.setdefault(category, [])
        temp_list = categories[category]
        temp_list.append(splitted_line[1:])
        categories[category] = temp_list

    return categories


# U1.txt task
file=open("U1.txt")
content = file.read().splitlines()

file.close()

content = content[1:]

categories = parse_categories(content)

categories_steps = []

for category, steps in categories.items():
    categories_steps.append(calculate_category_steps(category, steps))

write_results(categories_steps)