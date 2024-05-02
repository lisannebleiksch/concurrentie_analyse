import matplotlib.pyplot as plt

# Example dataset of companies and their foundation years
companies = ['Apple', 'Microsoft', 'Google', 'Amazon', 'Facebook', 'Tesla', 'Netflix', 'Airbnb', 'Uber', 'Spotify']
foundation_years = [1976, 1975, 1998, 1994, 2004, 2003, 1997, 2008, 2009, 2006]

# Create a scatter plot
plt.figure(figsize=(10, 6))
for i, (name, year) in enumerate(zip(companies, foundation_years)):
    plt.scatter(year, i, label=name, s=100)  # s is the size of the dot

# Add annotations to each point
for i, (name, year) in enumerate(zip(companies, foundation_years)):
    plt.text(year, i, f' {name}', verticalalignment='center')

# Improve the layout
plt.yticks(range(len(companies)), [''] * len(companies))  # Hide y ticks
plt.title('Company Foundation Years')
plt.xlabel('Year Founded')
plt.xticks()
plt.grid(axis='x', linestyle='--')

plt.tight_layout()
plt.show()
