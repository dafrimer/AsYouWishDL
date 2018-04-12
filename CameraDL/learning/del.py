all_numbers = []
while True:
    n = input("Enter a number: ")
    if n == 'done':
        break
    try:
        n = int(n)
        all_numbers.append(n)
    except ValueError:
        print("bad data")

print(max(all_numbers), min(all_numbers) )
