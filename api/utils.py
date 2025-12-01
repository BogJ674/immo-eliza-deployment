def calculate_values_from_postal_code(postal_code: int) -> dict:
    """
    Given a postal code, return a dictionary with calculated values such as
    median income and median price for the corresponding municipality, arrondissement, and province.
    """
    import pandas as pd
    from pathlib import Path

    # Define data paths
    data_dir = Path(__file__).parent / "data"
    postal_codes_path = data_dir / "postal-codes-belgium.csv"
    income_path = data_dir / "median_income_2023.csv"
    prices_path = data_dir / "FR_immo_statbel_commune_aggregated.csv"

    try:
        # Load postal codes data to get REFNIS codes
        postal_df = pd.read_csv(postal_codes_path, sep=';', encoding='utf-8-sig')

        # Find the row for this postal code
        postal_row = postal_df[postal_df['Postal Code'] == postal_code]
        if postal_row.empty:
            return {
                "median_income_mun": 0,
                "median_income_arr": 0,
                "median_income_prv": 0,
                "median_price_house": 0,
                "median_price_apartment": 0,
            }

        # Extract codes - use the first row if multiple matches
        postal_row = postal_row.iloc[0]

        # Convert codes to strings, handling NaN values
        if pd.notna(postal_row['Municipality code']):
            municipality_code = str(int(float(postal_row['Municipality code'])))
        else:
            municipality_code = None

        if pd.notna(postal_row['Arrondissement code']):
            arrondissement_code = str(int(float(postal_row['Arrondissement code'])))
        else:
            arrondissement_code = None

        if pd.notna(postal_row['Province code']):
            province_code = str(int(float(postal_row['Province code'])))
        else:
            province_code = None

        municipality_name = postal_row['Municipality name (Dutch)'] or postal_row['Municipality name (French)']

        # If we don't have a municipality code, we can't proceed
        if not municipality_code:
            return {
                "median_income_mun": 0,
                "median_income_arr": 0,
                "median_income_prv": 0,
                "median_price_house": 0,
                "median_price_apartment": 0,
            }

        # Load median income data
        income_df = pd.read_csv(income_path, encoding='utf-8-sig')
        income_df['municipality_upper'] = income_df['municipality_upper'].str.strip().str.upper()

        # Get median income for municipality
        municipality_name_upper = str(municipality_name).strip().upper()
        income_row = income_df[income_df['municipality_upper'] == municipality_name_upper]
        median_income_mun = float(income_row['median_income'].iloc[0]) if not income_row.empty else 0

        # For arrondissement and province income, we'll use aggregated values
        # For now, using municipality value as proxy (you can enhance this later)
        median_income_arr = median_income_mun  # TODO: Calculate actual arrondissement average
        median_income_prv = median_income_mun  # TODO: Calculate actual province average

        # Load property prices data
        prices_df = pd.read_csv(prices_path, encoding='utf-8-sig')

        # Match by REFNIS code (municipality code)
        price_row = prices_df[prices_df['refnis'].astype(str) == municipality_code]

        median_price_house = 0
        median_price_apartment = 0

        if not price_row.empty:
            price_row = price_row.iloc[0]
            median_price_house = float(price_row['median_price_house']) if pd.notna(price_row['median_price_house']) else 0
            median_price_apartment = float(price_row['median_price_apartment']) if pd.notna(price_row['median_price_apartment']) else 0

        return {
            "median_income_mun": median_income_mun,
            "median_income_arr": median_income_arr,
            "median_income_prv": median_income_prv,
            "median_price_house": median_price_house,
            "median_price_apartment": median_price_apartment,
        }

    except Exception as e:
        print(f"Error calculating values for postal code {postal_code}: {str(e)}")
        return {
            "median_income_mun": 0,
            "median_income_arr": 0,
            "median_income_prv": 0,
            "median_price_house": 0,
            "median_price_apartment": 0,
        }