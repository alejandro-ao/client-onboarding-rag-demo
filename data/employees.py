import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

def generate_employee_data(num_employees=5):
    employees = []
    for _ in range(num_employees):

        employee = {
            "employee_id": fake.uuid4(),
            "name": fake.first_name(),
            "lastname": fake.last_name(),
            "email": fake.email(),
            "phone_number": fake.phone_number(),
            "position": random.choice([
                "Research Scientist", 
                "Software Engineer", 
                "Operations Manager", 
                "HR Specialist", 
                "Security Officer"
            ]),
            "department": random.choice([
                "R&D", 
                "IT", 
                "Operations", 
                "HR", 
                "Security"
            ]),
            "skills": random.sample([
                "Python", "Project Management", "Data Analysis", 
                "Genetic Research", "Cybersecurity", "Machine Learning",
                "Leadership", "Database Management", "Public Speaking"
            ], k=random.randint(2, 5)),
            "location": random.choice([
                "Raccoon City HQ", 
                "Umbrella Europe", 
                "Umbrella Asia", 
                "Umbrella North America", 
                "Umbrella South America"
            ]),
            "hire_date": (
                datetime.now() - timedelta(days=random.randint(1, 365 * 10))
            ).strftime("%Y-%m-%d"),
            "supervisor": fake.name(),
            "salary": round(random.uniform(40000, 120000), 2),
        }

        employees.append(employee)

    return employees