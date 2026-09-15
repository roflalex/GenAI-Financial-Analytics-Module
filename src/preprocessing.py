"""
Data generation and cleaning pipeline for messy financial transactions.
Simulates real-world financial transaction data with deliberate dirtiness:
- Inconsistent date formats
- Mixed currency formats and negative/positive signs
- Dirty merchant text (store codes, extra whitespace, city tags, typos)
- Missing values and duplicate entries
- Seasonal spending trends and irregular large purchases / anomalies
"""

import os
import re
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Categories and realistic merchant templates
MERCHANT_TEMPLATES = {
    'Groceries': [
        'TESCO EXTRA {city}', 'Tesco Store #{code}', 'DUNNES STORES {city}', 'Lidl {code} {city}',
        'ALDI STORE {code}', 'SuperValu {city}', 'Marks & Spencer Food', 'SPAR EXPRESS {city}',
        'Tesco Express', 'LIDL IRELAND', 'ALDI IRELAND GMBH'
    ],
    'Dining & Takeaway': [
        'Starbucks Coffee #{code}', 'McDonalds {city}', 'DELIVEROO.IE *{code}', 'JUST EAT {code}',
        'Costa Coffee {city}', 'Boojum Burrito {city}', 'Nandos {city}', 'Gourmet Burger Kitchen',
        'THE LOCAL PUB & GRILL', 'Dominos Pizza {city}', 'Insomnia Coffee Co'
    ],
    'Transport': [
        'LEAP CARD AUTO TOPUP', 'DUBLIN BUS {code}', 'IRISH RAIL ONLINE', 'UBER *TRIP {code}',
        'CIRCLE K {city}', 'Applegreen Motorway Services', 'SHELL PETROL {code}', 'FREE NOW TAXI DUBLIN',
        'TFI Leap Card Topup', 'DART TICKET MACHINE'
    ],
    'Entertainment & Subscriptions': [
        'SPOTIFY AB {code} EUR', 'Netflix.com monthly', 'AMZN PRIME MEMBER', 'DISNEY PLUS MONTHLY',
        'PLAYSTATION NETWORK {code}', 'Steam Games Purchase', 'Audible UK Monthly', 'ODEON CINEMAS {city}',
        'YOUTUBE PREMIUM', 'APPLE.COM/BILL'
    ],
    'Shopping': [
        'AMZN Mktp UK*{code}', 'Amazon.de Purchase', 'ZARA IRELAND {city}', 'ASOS.COM ORDERS',
        'IKEA DUBLIN STORE', 'PENNEYS PRIMARK {city}', 'APPLE STORE RETAIL', 'BOOTS PHARMACY {city}',
        'Decathlon Sports Ireland', 'TK MAXX {city}', 'Brown Thomas Dublin'
    ],
    'Utilities & Bills': [
        'ELECTRIC IRELAND DD', 'Bord Gais Energy DD', 'VODAFONE BILL PAY', 'VIRGIN MEDIA IRELAND',
        'EIR TELECOM DIRECT DEBIT', 'Irish Water Direct Debit', 'Council Tax Payment', 'Sky Ireland TV & Broadband'
    ],
    'Health & Fitness': [
        'FLYEfit Gym Monthly DD', 'Gym Plus Membership', 'PureGym Direct Debit', 'Holland & Barrett {city}',
        'Boots Healthcare {city}', 'Dentist Clinic Dublin', 'Physiotherapy Care Dublin'
    ],
    'Travel': [
        'RYANAIR *{code} AIRLINE', 'AER LINGUS DUBLIN', 'BOOKING.COM HOTEL', 'AIRBNB * {code}',
        'HOTELS.COM RESERVATION', 'Dublin Airport Parking', 'DAA DUTY FREE'
    ]
}

CATEGORY_AMOUNT_DISTRIBUTIONS = {
    'Groceries': {'mean': 52.0, 'std': 28.0, 'min': 8.50, 'max': 185.0},
    'Dining & Takeaway': {'mean': 24.0, 'std': 18.0, 'min': 3.80, 'max': 110.0},
    'Transport': {'mean': 18.0, 'std': 16.0, 'min': 2.40, 'max': 95.0},
    'Entertainment & Subscriptions': {'mean': 14.5, 'std': 12.0, 'min': 4.99, 'max': 79.99},
    'Shopping': {'mean': 68.0, 'std': 55.0, 'min': 10.0, 'max': 380.0},
    'Utilities & Bills': {'mean': 95.0, 'std': 45.0, 'min': 35.0, 'max': 240.0},
    'Health & Fitness': {'mean': 38.0, 'std': 22.0, 'min': 12.0, 'max': 120.0},
    'Travel': {'mean': 180.0, 'std': 130.0, 'min': 25.0, 'max': 750.0}
}

CITIES = ['DUBLIN', 'CORK', 'GALWAY', 'LIMERICK', 'BRAY', 'SWORDS', 'DUNDALK', 'KILKENNY']


def generate_messy_transactions(n_transactions: int = 3600, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic transaction dataset covering ~2.5 years,
    injected with real-world messiness:
    - Multiple date formats
    - Irregular casing, typos, store identifiers
    - Negative and positive amounts with currency glyphs
    - Occasional missing values and duplicate records
    """
    random.seed(seed)
    np.random.seed(seed)

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 6, 30)
    total_days = (end_date - start_date).days

    rows = []
    categories = list(MERCHANT_TEMPLATES.keys())

    # Category weights (monthly transaction frequency)
    weights = [0.32, 0.22, 0.16, 0.10, 0.10, 0.04, 0.04, 0.02]

    for _ in range(n_transactions):
        cat = random.choices(categories, weights=weights)[0]
        template = random.choice(MERCHANT_TEMPLATES[cat])

        # Fill template tokens
        city = random.choice(CITIES)
        code = str(random.randint(1000, 9999))
        raw_merchant = template.format(city=city, code=code)

        # Random timestamp
        day_offset = random.randint(0, total_days)
        tx_date = start_date + timedelta(days=day_offset)

        # Amount generation from normal/truncated distributions
        dist = CATEGORY_AMOUNT_DISTRIBUTIONS[cat]
        amount_val = max(dist['min'], min(dist['max'], float(np.random.normal(dist['mean'], dist['std']))))

        # Seasonal spending boost in December (Christmas shopping & dining)
        if tx_date.month == 12 and cat in ['Shopping', 'Dining & Takeaway', 'Groceries']:
            amount_val *= random.uniform(1.25, 1.65)

        # Weekend spending boost on dining and entertainment
        is_weekend = tx_date.weekday() >= 5
        if is_weekend and cat in ['Dining & Takeaway', 'Entertainment & Subscriptions']:
            amount_val *= random.uniform(1.15, 1.4)

        # Deliberate occasional extreme anomaly (~1.5% chance)
        if random.random() < 0.015:
            if cat == 'Shopping':
                amount_val = round(random.uniform(450.0, 980.0), 2)  # Tech purchase / furniture
            elif cat == 'Dining & Takeaway':
                amount_val = round(random.uniform(220.0, 480.0), 2)  # Group party / fine dining
            elif cat == 'Travel':
                amount_val = round(random.uniform(650.0, 1450.0), 2)  # International flights

        amount_val = round(amount_val, 2)

        # Apply realistic text noise
        noise_type = random.random()
        if noise_type < 0.15:
            raw_merchant = raw_merchant.lower()
        elif noise_type < 0.30:
            raw_merchant = f"  {raw_merchant}   "  # leading/trailing spaces
        elif noise_type < 0.38:
            raw_merchant = raw_merchant.title()

        # Date formatting variety
        date_format_choice = random.random()
        if date_format_choice < 0.55:
            raw_date_str = tx_date.strftime('%Y-%m-%d')
        elif date_format_choice < 0.80:
            raw_date_str = tx_date.strftime('%d/%m/%Y')
        elif date_format_choice < 0.92:
            raw_date_str = tx_date.strftime('%b %d, %Y')
        else:
            raw_date_str = tx_date.strftime('%Y.%m.%d')

        # Amount formatting variety (negative expenses, currency symbols, euro signs)
        amt_fmt_choice = random.random()
        if amt_fmt_choice < 0.45:
            raw_amount_str = f"-{amount_val:.2f}"
        elif amt_fmt_choice < 0.70:
            raw_amount_str = f"-€{amount_val:.2f}"
        elif amt_fmt_choice < 0.85:
            raw_amount_str = f"€{amount_val:.2f}"
        elif amt_fmt_choice < 0.95:
            raw_amount_str = f"{amount_val:.2f}"
        else:
            raw_amount_str = f"{amount_val:.2f} EUR"

        rows.append({
            'raw_date': raw_date_str,
            'raw_merchant': raw_merchant,
            'raw_amount': raw_amount_str,
            'true_category': cat,
            'true_date': tx_date.strftime('%Y-%m-%d'),
            'true_amount': amount_val
        })

    df = pd.DataFrame(rows)

    # Inject missing values
    # ~2.5% missing merchant
    missing_merchant_idx = df.sample(frac=0.025, random_state=seed).index
    df.loc[missing_merchant_idx, 'raw_merchant'] = np.nan

    # ~1.5% missing amount
    missing_amount_idx = df.sample(frac=0.015, random_state=seed + 1).index
    df.loc[missing_amount_idx, 'raw_amount'] = np.nan

    # Inject duplicate rows (~2% duplicates)
    dup_rows = df.sample(frac=0.02, random_state=seed + 2)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Shuffle rows to simulate raw unsorted log
    df = df.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)
    return df


def parse_inconsistent_date(date_val):
    """
    Robust parser handling multiple date string formats into standard pd.Timestamp.
    """
    if pd.isna(date_val):
        return pd.NaT

    if isinstance(date_val, (pd.Timestamp, datetime)):
        return pd.to_datetime(date_val)

    s = str(date_val).strip()

    formats = [
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%b %d, %Y',
        '%B %d, %Y',
        '%Y.%m.%d',
        '%d-%m-%Y',
        '%m/%d/%Y'
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(datetime.strptime(s, fmt))
        except (ValueError, TypeError):
            continue

    try:
        return pd.to_datetime(s, errors='coerce')
    except Exception:
        return pd.NaT


def parse_inconsistent_amount(amt_val) -> float:
    """
    Cleans messy currency strings (e.g. '-€48.72', '48.72 EUR', '-50.00')
    and extracts a standardized positive float representing expense amount.
    """
    if pd.isna(amt_val):
        return np.nan

    if isinstance(amt_val, (int, float)):
        return abs(float(amt_val))

    s = str(amt_val).strip()
    s_cleaned = re.sub(r'[€$£?A-Za-z\s,]', '', s)

    try:
        val = float(s_cleaned)
        return abs(val)
    except ValueError:
        return np.nan


def clean_merchant_text(text: str) -> str:
    """
    Normalizes dirty merchant text by:
    - Stripping store codes (#1234, *TRIP, etc.)
    - Removing transaction boilerplate (DD, Direct Debit, Purchase, Store)
    - Normalizing casing and excess spaces
    """
    if pd.isna(text):
        return 'UNKNOWN'

    t = str(text).strip().upper()

    t = re.sub(r'#\d+', '', t)
    t = re.sub(r'\*TRIP\s*\d+', '', t)
    t = re.sub(r'\*[\dA-Z]+', '', t)
    t = re.sub(r'\d{4,}', '', t)
    t = re.sub(r'(EUR|DD|DIRECT DEBIT|MONTHLY|PURCHASE|ORDER|ONLINE|STORE|MEMBER)', '', t)
    t = re.sub(r'[*\-_/.,#:]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()

    return t if t else 'UNKNOWN'


def clean_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end cleaning pipeline:
    1. Removes duplicate rows
    2. Drops or imputes records with missing critical fields
    3. Normalizes dates into standard datetime64
    4. Normalizes amounts into positive numerical floats
    5. Cleans and standardizes merchant descriptions
    6. Adds calendar features (year, month, year_month, day_of_week, is_weekend)
    """
    df = raw_df.copy()

    initial_len = len(df)
    df = df.drop_duplicates(subset=['raw_date', 'raw_merchant', 'raw_amount']).reset_index(drop=True)

    df = df.dropna(subset=['raw_amount']).copy()
    df['raw_merchant'] = df['raw_merchant'].fillna('Unknown Merchant')

    df['date'] = df['raw_date'].apply(parse_inconsistent_date)
    df = df.dropna(subset=['date']).copy()
    df['date'] = pd.to_datetime(df['date'])

    df['amount'] = df['raw_amount'].apply(parse_inconsistent_amount)
    df = df.dropna(subset=['amount']).copy()
    df = df[df['amount'] > 0.0].copy()

    df['clean_merchant'] = df['raw_merchant'].apply(clean_merchant_text)

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    df['day_of_week'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.dayofweek >= 5

    if 'true_category' in df.columns:
        df['category'] = df['true_category']
    else:
        df['category'] = 'Uncategorised'

    df = df.sort_values('date').reset_index(drop=True)

    final_len = len(df)
    print(f"Cleaning complete: {initial_len} raw records -> {final_len} cleaned records.")
    return df


def generate_and_save_data(raw_path: str = 'data/raw/messy_transactions.csv',
                           clean_path: str = 'data/processed/cleaned_transactions.csv'):
    """Generates raw synthetic messy data and processes it, saving both to disk."""
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)

    print("Generating realistic messy transaction dataset...")
    raw_df = generate_messy_transactions(n_transactions=3600, seed=42)
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved raw messy transactions to {raw_path} ({len(raw_df)} rows)")

    print("Cleaning transactions...")
    clean_df = clean_transactions(raw_df)
    clean_df.to_csv(clean_path, index=False)
    print(f"Saved cleaned transactions to {clean_path} ({len(clean_df)} rows)")
    return raw_df, clean_df


if __name__ == '__main__':
    generate_and_save_data()
