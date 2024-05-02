import pandas as pd

from utils.utils import (download_excel, read_excel, df_shared_customers, plot_shared_customers,
                         plot_customer_count, plot_domain_popularity, plot_servicemap, plot_omzet,
                         read_total_excel, plot_post, plot_followers, plot_werknemers, plot_founded,
                         plot_domain_popularity_smaller, plot_dienstaanbod)


def main():
    url_klant = 'https://docs.google.com/spreadsheets/d/1gSpYB7s5r8G5xQ-tplqeSYcXtlp967COCq1qdAW2a2o/export?format=xlsx'  # Replace with your Google Sheets download link
    url_total = "https://docs.google.com/spreadsheets/d/1TZEWvY2vpXRE3QVrmlO7e1JIDNX_XfGiJQ64qW2IQYQ/export?format=xlsx"
    excel_file_klant = download_excel(url_klant)
    int_df_klant, string_df_klant = read_excel(excel_file_klant)
    excel_file_total = download_excel(url_total)
    df_total = read_total_excel(excel_file_total)

    plot_servicemap(pd.read_excel(url_total))
    df_shared_cust = df_shared_customers(string_df_klant)
    plot_domain_popularity(int_df_klant)
    plot_domain_popularity_smaller(int_df_klant)
    plot_customer_count(int_df_klant)
    plot_shared_customers(df_shared_cust)
    plot_omzet(df_total)
    plot_post(df_total)
    plot_followers(df_total)
    plot_werknemers(df_total)
    plot_founded(df_total)
    # plot_dienstaanbod(df_total)







if __name__ == "__main__":
    main()