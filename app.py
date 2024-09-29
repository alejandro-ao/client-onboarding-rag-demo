from data.employees import generate_employee_data
import json

if __name__ == "__main__":

    # Here is an example of how to use the get_user_data function
    users = generate_employee_data(1)[0]
    print("\n\nUser data:")
    print(json.dumps(users, indent=4))
