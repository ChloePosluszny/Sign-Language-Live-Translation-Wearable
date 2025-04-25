# Load names from txt file
def load_names(file_path):
    with open(file_path, 'r') as f:
        names = set(line.strip().lower() for line in f)
    return names

# Function to insert spaces when names are detected
def insert_spaces_for_names(text, names_set):
    idx = 0
    result = ""
    while idx < len(text):
        matched = False
        for name in sorted(names_set, key=len, reverse=True):  # Match longest names first
            if text.startswith(name, idx):
                result += name + " "
                idx += len(name)
                matched = True
                break
        if not matched:
            result += text[idx]
            idx += 1
    return result.strip()

# Example usage
names_set = load_names('names.txt')

input_text = "hello davidcassitymy name is rahmankhandakar"

output = insert_spaces_for_names(input_text.lower(), names_set)
print(output.replace("  ", " "))

