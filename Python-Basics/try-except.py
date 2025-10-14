while True:
    user_input = input("Enter a number: ")
    try:
        number = int(user_input)
        break  # Exit the loop if the input is valid
    
    except ValueError:
        print("Invalid input. Please enter a numerical value.")

print("Entered no. is:",number)