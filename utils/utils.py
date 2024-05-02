from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import pandas as pd
import seaborn as sns
import networkx as nx
import seaborn as sns

def download_excel(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        raise Exception(f"Failed to download file: Status code {response.status_code}")

def read_excel(file):
    # Read the Excel file, considering the first two rows as header
    df = pd.read_excel(file)
    df_counts = df.fillna(0)
    # Preprocessing: Counting the number of customers, handling both strings and non-string cells
    for col in df_counts.columns[1:]:  # Skipping the 'Domein' column
        df_counts[col] = df_counts[col].apply(lambda x: len(x.split(', ')) if isinstance(x, str) else 0)
    return df_counts, df

def read_total_excel(file):
    df = pd.read_excel(file)
    # Transponeer de dataframe zodat bedrijven de index worden en er slechts één kolom met omzetwaarden is
    df_transposed = df.transpose()
    # Reset index to move the current index to a column
    df_transposed = df_transposed.reset_index()

    # Set the column names to the values in the first row
    df_transposed.columns = df_transposed.iloc[0]
    # Drop the first row, which contains the unwanted column names
    df_transposed = df_transposed.drop(0)
    # Reset index to make it start from 0 again
    df_transposed = df_transposed.reset_index(drop=True)

    return df_transposed

def df_shared_customers(df):
    # Initialize a dictionary to store customers and the companies that have them
    customer_dict = {}
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        companies = row.index[1:]  # Skip the first column ('Domein')
        for company in companies:
            customers = row[company]
            if isinstance(customers, str):
                customers = customers.split(', ')
                for customer in customers:
                    # Convert customer name to lowercase for case-insensitive comparison
                    customer = customer.lower()
                    if customer not in customer_dict:
                        customer_dict[customer] = [company]
                    else:
                        customer_dict[customer].append(company)

    # Initialize an empty list to store tuples of customer and companies
    results = []

    # Iterate over the customer dictionary to group companies sharing the same customers
    for customer, companies in customer_dict.items():
        if len(companies) > 1:
            results.append((customer, ', '.join(companies)))

    # Create DataFrame from results
    results_df = pd.DataFrame(results, columns=['Customer', 'Companies'])
    return results_df

def plot_shared_customers(df):
    # Split the 'Companies' string into individual companies for each customer
    # print(df)
    edges = []
    for index, row in df.iterrows():
        companies = row['Companies'].split(', ')
        for company in companies:
            edges.append((row['Customer'], company))

    # Create a network graph
    G = nx.Graph()
    G.add_edges_from(edges)
    # Identify unique companies and customers
    unique_companies = set([company for sublist in df['Companies'].str.split(', ') for company in sublist])
    unique_customers = set(df['Customer'])

    emphasize_company = 'Future Facts'
    # Create a color map based on node type
    color_map = []
    for node in G:
        if node == emphasize_company:
            color_map.append('dodgerblue')  # Color for companies
        elif node in unique_companies:
            color_map.append('lightblue')  # Color for companies
        else:
            color_map.append('lightgreen')  # Color for customers


    # Plot the updated network graph with different colors for companies and customers
    plt.figure(figsize=(20, 16))
    nx.draw(G, with_labels=True, node_color=color_map, edge_color='gray', font_size=20, node_size=5500, alpha=0.5, font_color='black', font_weight='bold') # Set alpha value here
    # plt.title('Network Graph of Customers and Their Shared Companies (Differentiated)')
    plt.savefig('plots/network_graph.png')

def plot_customer_count(df):
    plt.figure(figsize=(18, 14))
    company_customer_counts = df.drop('Domein', axis=1).sum().sort_values()
    bars=company_customer_counts.plot(kind='bar', color='orange')
    # plt.title('Total Customer Counts per Company')
    plt.ylabel('Totaal klanten op de website', fontsize=20)
    # plt.xlabel('Company', fontsize=20)
    # Tekst bovenop de balken toevoegen
    for index, value in enumerate(company_customer_counts):
        plt.text(index, value, f'{int(value)}', ha='center', va='bottom', fontsize=20)

    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig('plots/customer_counts.png')

def plot_domain_popularity(df, emphasis_column='Future Facts'):
    # Creating a new DataFrame with both the emphasis column and the total
    df_emphasis = df[['Domein', emphasis_column]].copy()
    df_emphasis['Total'] = df.drop('Domein', axis=1).sum(axis=1)
    df_emphasis = df_emphasis.sort_values(by='Total', ascending=False)

    # Plotting the grouped bar plot
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    df_emphasis.set_index('Domein').plot(kind='bar', stacked=True, ax=ax)
    plt.ylabel('Total Customers', fontsize=20)
    plt.xlabel('Domain', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.tight_layout()
    plt.legend(fontsize=14)
    plt.savefig('plots/domain_popularity.png')

def plot_domain_popularity_smaller(df, emphasis_column='Future Facts',
                                   selected_columns=["Vantage AI", "Cmotions", "ADC", "Future Facts",
                                                     "Bigdata Republic", "Xomnia", "ADC", "Data Science Lab",
                                                     "Infosupport (nl)", "Digital Power"]):
    # Creating a new DataFrame with the selected columns and the emphasis column
    df_selected = df[['Domein'] + selected_columns].copy()

    # Filter out rows where the emphasis column is zero
    # df_selected = df_selected[df_selected[emphasis_column] > 0]

    # Creating a new DataFrame with the emphasis column and the total
    df_emphasis = df_selected.copy()
    df_emphasis['Total'] = df_selected.drop('Domein', axis=1).sum(axis=1)

    # Sort by total but do not include the total in the plot
    df_emphasis = df_emphasis.sort_values(by='Total', ascending=False).drop('Total', axis=1)

    # Plotting the grouped bar plot
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    # Use the colormap of your choice, e.g., 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    df_emphasis.set_index('Domein').plot(kind='bar', stacked=True, ax=ax, colormap='tab20c')  # Example with 'viridis'
    plt.ylabel('Total Customers', fontsize=20)
    plt.xlabel('Domain', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.tight_layout()
    plt.legend(fontsize=14)
    plt.savefig('plots/domain_popularity_smaller.png')

def plot_servicemap(df):
    # Selecteren van de rij 'dienstenaanbod' voor de heatmap
    diensten_row = df.loc[df['bedrijf'] == 'dienstenaanbod', :].drop('bedrijf', axis=1)

    # Opnieuw splitsen van de diensten per bedrijf en omzetten naar een lijst
    diensten_per_bedrijf = diensten_row.iloc[0].str.split(', ')

    # Creëren van een nieuwe DataFrame voor de heatmap, gebaseerd op de nieuwe data
    heatmap_df = pd.DataFrame(index=diensten_row.columns,
                                  columns=pd.unique(diensten_row.iloc[0].str.cat(sep=', ').split(', ')))

    # Initialiseren van de DataFrame met 0
    heatmap_df.fillna(0, inplace=True)

    # Invullen van de DataFrame: zet een cel op 1 als het bedrijf de dienst aanbiedt
    for bedrijf in diensten_per_bedrijf.index:
        for dienst in diensten_per_bedrijf[bedrijf]:
            heatmap_df.at[bedrijf, dienst] = 1

    # Genereren van de heatmap met de nieuwe data
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_df.T, cmap="Blues", linewidths=.5, annot=True, fmt="d", cbar=False)
    # plt.title('Overzicht van Diensten per Bedrijf (Uitgebreid)')
    # plt.xlabel('Bedrijven')
    # plt.ylabel('Diensten')
    plt.xticks(rotation=45, ha="right", fontsize=18)
    plt.yticks(rotation=0, ha="right", fontsize=18)
    plt.tight_layout()

    # Tonen van de nieuwe heatmap
    plt.savefig('plots/servicemap.png')

def omzet_naar_num(omzet_str):
    omzet_mapping = {
        '2.5-5 mil': 5000000,
        '10-20 mil': 10000000,
        '20-50 mil': 20000000,
        '500-1 bil': 510000000,
        '1 bil +': 1100000000,
        '200 mil': 200000000,
    }
    # Voor eenvoudige gevallen waar de omzet direct gemapt kan worden
    if omzet_str in omzet_mapping:
        return omzet_mapping[omzet_str]
    # Voor omzetten met 'k' (duizenden) of directe getallen
    try:
        if 'k' in omzet_str:
            return float(omzet_str.replace('k', '').strip()) * 1000  # Omzetten naar doesoe
        return float(omzet_str.strip())
    except:
        return 0  # Terugvallen naar 0 als de omzet niet geparsed kan worden

def plot_omzet(df):
    # Aannemend dat 'omzet_naar_num' reeds gedefinieerd is en correct werkt.
    df['omzet_num'] = df['omzet'].apply(omzet_naar_num)

    # Sorteren van de bedrijven op basis van hun omgezette omzet
    df = df.sort_values(by='omzet_num', ascending=True)

    # Genereren van de bar chart met numerieke omzetwaarden
    plt.figure(figsize=(14, 8))
    # Aanpassen van de waarde in de kolom 'bedrijf' van 'Eraneos (nl)' naar 'Eraneos Group'
    df['bedrijf'] = df['bedrijf'].replace('Eraneos (nl)', 'Eraneos Group')
    bars = plt.barh(df['bedrijf'], df['omzet_num'], color='teal')
    plt.xscale('log')  # Logaritmische schaal voor de x-as
    plt.xlabel('Omzet in miljoenen [€]', fontsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Loop over de bars en voeg de omzetrange tekst toe aan het einde van elke bar
    for bar, omzet_range in zip(bars, df['omzet']):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2., f'{omzet_range}', ha='left', va='center', fontsize=16)

    plt.tight_layout()

    # Tonen van de bar chart
    plt.savefig('plots/omzet.png')

def convert_to_days(s):
    time_units = {'s': 1/86400, 'm': 1/1440, 'h': 1/24, 'd': 1, 'w': 7, 'y': 365}  # Eenheidswaarden nu in dagen
    total_days = 0
    current_number = ''
    for char in s:
        if char.isdigit():
            current_number += char
        elif char in time_units:
            total_days += int(current_number) * time_units[char]
            current_number = ''
    return total_days

# Aangepaste plot functie
def plot_post(df):
    df['last_post_days'] = df['last post (21 feb)'].apply(convert_to_days)
    # Data sorteren op 'last_post_days'
    df_sorted = df.sort_values(by='last_post_days', ascending=True)
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df_sorted['bedrijf'], df_sorted['last_post_days'], color='royalblue')
    plt.yscale('log')  # Logaritmische schaal voor de y-as
    plt.ylabel('Activiteit', fontsize=20)
    plt.xticks(rotation=45, fontsize=20)  # Draai de x-as labels voor betere leesbaarheid
    plt.yticks(fontsize=20)
    for bar, followers_text in zip(bars, df_sorted['last post (21 feb)']):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{followers_text}',
                 ha='center', va='bottom', fontsize=20)
    plt.tight_layout()
    plt.savefig("plots/last_post_times.png")


def plot_followers(df):
    df['followers'] = df['followers linkedin'].apply(omzet_naar_num)
    # Data sorteren op 'last_post_days'
    df_sorted = df.sort_values(by='followers', ascending=True)
    plt.figure(figsize=(14, 8))  # Vergroot indien nodig
    bars = plt.bar(df_sorted['bedrijf'], df_sorted['followers'], color='royalblue')
    plt.yscale('log')  # Logaritmische schaal voor de y-as
    plt.ylabel('Aantal volgers Linkedin', fontsize=20)
    plt.xticks(rotation=45, fontsize=20)  # Aanpassing voor leesbaarheid
    plt.yticks(fontsize=20)
    for bar, followers_text in zip(bars, df_sorted['followers linkedin']):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{followers_text}',
                 ha='center', va='bottom', fontsize=20)
    plt.subplots_adjust(bottom=0.25)  # Pas de onderste marge aan
    plt.tight_layout()
    plt.savefig("plots/followers_linkedin.png")


def plot_werknemers(df):
    # Data sorteren op 'associated member'
    df_sorted = df.sort_values(by='associated member', ascending=True)

    # Plotting de gesorteerde data
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df_sorted['bedrijf'], df_sorted['associated member'], color='royalblue')

    # Logaritmische schaal voor de y-as
    plt.yscale('log')

    # Labels en titels (optioneel, uitgecommentarieerd)
    # plt.title('Laatste Post Tijden per Bedrijf', fontsize=20)
    # plt.xlabel('Bedrijf', fontsize=16)
    plt.ylabel('Associated members Linkedin', fontsize=20)
    plt.xticks(rotation=45, fontsize=20)  # Draai de x-as labels voor betere leesbaarheid
    plt.yticks(fontsize=16)

    # Tekst bovenop de balken toevoegen
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom', fontsize=16)

    plt.tight_layout()
    plt.savefig("plots/werknemers_linkedin.png")

def plot_founded(df):
    companies = df['bedrijf']
    foundation_years = df['established linkedin']
    # Create a scatter plot
    plt.figure(figsize=(10, 6))

    # Determine the range for the x-axis
    min_year = min(foundation_years) - 5  # 5-year buffer on the earlier side
    max_year = max(foundation_years) + 5  # 5-year buffer on the later side

    # Create a scatter plot
    plt.figure(figsize=(12, 8))  # Adjust figure size for better visibility
    for i, (name, year) in enumerate(zip(companies, foundation_years)):
        plt.scatter(year, i, label=name, s=120)  # s is the size of the dot

    # Add annotations to each point
    for i, (name, year) in enumerate(zip(companies, foundation_years)):
        plt.text(year, i + 0.4, f' {name}', verticalalignment='center', fontsize=20)

    # Set the x and y axis limits
    plt.xlim(min_year, max_year+ 2)
    plt.ylim(-1, len(companies))  # Extend y-axis to provide space at the top and bottom

    # Improve the layout
    plt.yticks(range(len(companies)), [''] * len(companies))  # Hide y ticks
    # plt.title('Company Foundation Years')
    plt.xlabel('Year Founded', fontsize=20)
    plt.xticks(fontsize=20)
    plt.grid(axis='x', linestyle='--')

    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/founded_linkedin.png")



def plot_dienstaanbod(df):
    diensten = df["dienstenaanbod"]

    diensten = set()
    for lst in df.dienstenaanbod:
        diensten.update(lst)
    diensten = sorted(diensten)

    # Initialiseren van de matrix
    matrix = pd.DataFrame(0, index=df.bedrijf, columns=diensten)

    # Vullen van de matrix
    for index, row in df.iterrows():
        for dienst in row['dienstenaanbod']:
            matrix.at[row['bedrijf'], dienst] = 1

    # Tekenen van de heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, cmap="YlGnBu", annot=True, cbar=False)
    plt.title('Dienstaanbod per Bedrijf')
    plt.ylabel('Bedrijf')
    plt.xlabel('Dienst')
    plt.show()