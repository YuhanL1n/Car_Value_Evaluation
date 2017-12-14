# Developed by Chu-Sheng Ku

import pandas as pd
import matplotlib.pyplot as plt


cars = pd.read_csv('cars.csv')
# Select the cars made from 1988 to 2018
cars = cars.loc[cars['year'].isin(range(1988, 2019))]
cars.info()

# Group the price of car by year
price_groupby_year = cars['price'].groupby(cars['year'])
price_groupby_year.describe()

year = []
price = []
count = []
# Remove the outliers by price
for name, group in price_groupby_year:
    q1, q3 = group.quantile([0.25, 0.75])
    iqr = q3 - q1
    group = group[(group > q1 - iqr * 1.5) & (group < q3 + iqr * 1.5)]
    year.append(name)
    price.append(group.mean())
    count.append(group.size)

# Plot the scatter chart of year and the mean of price
plt.scatter(year, price)
plt.title('The Correlation of Price and Year')
plt.xlabel('Year')
plt.ylabel('Price')
plt.grid()    
plt.show()

# Plot the bar chart of year and count
plt.bar(year, count)
plt.title('Distribution of Cars Made from 1988 to 2018 (N~180K)')
plt.xlabel('Year')
plt.ylabel('Number of Cars Posted on Craigslist')
plt.show()

# Print out the cars made in 2018 to see why the mean of price is not reasonable
print(cars.loc[(cars['year'] == 2018) & (cars['price'] < 1000)])

