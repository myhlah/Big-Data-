import pandas as pd
import numpy as np

# Generate 1500 random numbers from 0 to 4
random_numbers = np.random.randint(0, 5, size=1500)

# Create a DataFrame to format the data into a table
random_numbers_df = pd.DataFrame(random_numbers, columns=['Random Numbers'])

# Display the first 10 rows of the DataFrame
random_numbers_df.to_csv('random_numbers.csv', index=False)

print("Random numbers generated and saved as 'random_numbers.csv'")