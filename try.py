from collections import Counter

input_string = """
Some introduction text
More irrelevant lines

Iteration 1:
    Output: J. grasp spoon
Iteration 2:
    Output: L. move to blue bowl
Iteration 3:
    Output: A. scoop
Iteration 4:
    Output: M. move to white bowl
"""

result = {}
# Split input into lines and process each iteration
lines = input_string.strip().split("\n")
iteration = None
processing = False  # Flag to indicate when to start parsing

for line in lines:
    line = line.strip()
    
    if line.startswith("Iteration"):
        processing = True  # Start processing once "Iteration" is found
        iteration = int(line.split()[1].strip(":"))
    
    elif processing and line.startswith("Output:") and iteration is not None:
        content = line.split("Output:")[1].strip().split(". ")[0]
        char_counts = Counter(content)
        if iteration in result:
            for char, count in char_counts.items():
                result[iteration][char] = result[iteration].get(char, 0) + count
        else:
            result[iteration] = dict(char_counts)

print(result)
