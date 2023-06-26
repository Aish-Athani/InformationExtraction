import bert_classification

print()
print("Welcome to the event extracter! Please choose one of the following modes to enter your input: ")
print("(1) Extract information from a single passage through console")
print("(2) Extract information from a multiple files from the Resources directory")

user_input = input()
if user_input == '1':
    print("Please enter enter the text and press EOF to finish\n")
    contents = []
    line = ""
    while line != "EOF":
        try:
            line = input()
        except EOFError:
            break
        if line != "EOF":
            contents.append(line)

    f = open("example.txt","w")
    for line in contents:
        f.write("%s\n" % line)
    f.close()

    # bert_classification.get_terminal_output()
else:
    directory = "Resources/TerrorismEventData/test-doc" 
    # print("Please enter enter the directory: \n")
    # directory = input().strip()
    bert_classification.get_file_output(directory)