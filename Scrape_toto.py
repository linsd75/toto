
import pandas as pd
import requests
import time
import os
from datetime import datetime
import re
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def solve_winning_numbers(s):
    """
    Parses a string of concatenated digits into 6 sorted numbers (1-49).
    Uses backtracking to resolve ambiguity.
    """
    s = str(s).strip()
    if not s.isdigit():
        return []
        
    solutions = []

    def backtrack(idx, current_nums):
        if idx == len(s):
            if len(current_nums) == 6:
                solutions.append(list(current_nums))
            return

        if len(current_nums) >= 6:
            return

        # Try 1 digit
        if idx + 1 <= len(s):
            n1 = int(s[idx:idx+1])
            is_valid = (1 <= n1 <= 49)
            is_increasing = (len(current_nums) == 0) or (n1 > current_nums[-1])
            
            if is_valid and is_increasing:
                backtrack(idx + 1, current_nums + [n1])
                if solutions: return

        # Try 2 digits
        if idx + 2 <= len(s):
            n2 = int(s[idx:idx+2])
            is_valid = (1 <= n2 <= 49)
            is_increasing = (len(current_nums) == 0) or (n2 > current_nums[-1])
            
            if is_valid and is_increasing:
                backtrack(idx + 2, current_nums + [n2])
                if solutions: return

    backtrack(0, [])
    # Return the first valid solution found
    return solutions[0] if solutions else []

def clean_scraped_data(df):
    """
    Cleans the scraped TOTO data by parsing winning numbers.
    Returns a cleaned DataFrame with separate columns for each winning number.
    """
    print("\nCleaning scraped data...")
    
    # Filter garbage rows
    if 'Draw' in df.columns:
        df = df[df['Draw'] != 'Draw']
    
    if 'Winning No.' not in df.columns:
        print("Column 'Winning No.' not found. Skipping cleaning.")
        return df

    # Process
    new_rows = []
    
    for _, row in df.iterrows():
        raw_win = row['Winning No.']
        parsed_nums = solve_winning_numbers(raw_win)
        
        if len(parsed_nums) == 6:
            # Create a dictionary for the new row
            new_row = row.to_dict()
            del new_row['Winning No.'] # Remove the messy column
            
            # Add separate columns
            for i, num in enumerate(parsed_nums):
                new_row[f'Win_{i+1}'] = num
            
            new_rows.append(new_row)

    clean_df = pd.DataFrame(new_rows)
    
    # Reorder columns to put Win_1...Win_6 near the front
    cols = clean_df.columns.tolist()
    win_cols = [f'Win_{i+1}' for i in range(6)]
    
    # Base columns (we want to keep 'Draw', 'Date', then winning numbers, then 'Addl No.')
    desired_order = []
    for c in ['Draw', 'Date']:
        if c in cols: desired_order.append(c)
    
    desired_order.extend(win_cols)
    
    if 'Addl No.' in cols:
        desired_order.append('Addl No.')
        
    # Append rest of columns
    for c in cols:
        if c not in desired_order:
            desired_order.append(c)
            
    clean_df = clean_df[desired_order]
    
    print(f"Cleaned {len(clean_df)} valid rows from {len(df)} original rows.")
    
    return clean_df

def get_latest_draw_from_file(filepath):
    """
    Reads a CSV file and returns the latest draw number.
    Returns None if file doesn't exist or is invalid.
    """
    try:
        df = pd.read_csv(filepath)
        if 'Draw' in df.columns and len(df) > 0:
            # Remove non-numeric rows and get max draw number
            df['Draw'] = pd.to_numeric(df['Draw'], errors='coerce')
            latest_draw = df['Draw'].max()
            return int(latest_draw) if not pd.isna(latest_draw) else None
    except (FileNotFoundError, pd.errors.EmptyDataError, Exception):
        pass
    return None

def backup_old_files(current_filename, current_directory):
    """
    Backs up any existing ToTo-*.csv files (except the current one) by renaming them to *.CSVBAK
    """
    import glob
    pattern = os.path.join(current_directory, "ToTo-*.csv")
    existing_files = glob.glob(pattern)
    
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        # Don't backup the current file
        if filename != current_filename:
            backup_name = filepath + "BAK"
            try:
                # If backup already exists, remove it first
                if os.path.exists(backup_name):
                    os.remove(backup_name)
                os.rename(filepath, backup_name)
                print(f"Backed up old file: {filename} -> {filename}BAK")
            except Exception as e:
                print(f"Warning: Could not backup {filename}: {e}")

def scrape_single_page(page, base_url, latest_draw=None):
    """
    Scrapes a single page and returns the data if found.
    Returns (page_number, dataframe) or (page_number, None) if no data found.
    """
    url = base_url.format(page)
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return (page, None, f"Status code: {response.status_code}")

        # Preserve multi-number cells by converting <br> tags into a sentinel token
        html_text = re.sub(r'(?i)<br\s*/?>', ' __BR__ ', response.text)
        dfs = pd.read_html(StringIO(html_text))
        
        if not dfs:
            return (page, None, "No tables found")
        
        # Find the correct table
        for df in dfs:
            if len(df.columns) > 5:  # Lottery table
                # If doing incremental update, filter old data
                if 'From Last' in df.columns:
                    def split_compact_numbers(s):
                        # Return list of 2-4 numbers parsed from a compact digit string, or None
                        if not s.isdigit():
                            return None

                        solutions = []

                        def backtrack(idx, current):
                            if idx == len(s):
                                if 2 <= len(current) <= 4:
                                    solutions.append(list(current))
                                return
                            if len(current) >= 4:
                                return

                            for width in (1, 2):
                                if idx + width <= len(s):
                                    n = int(s[idx:idx + width])
                                    if 1 <= n <= 49:
                                        backtrack(idx + width, current + [n])

                        backtrack(0, [])
                        if not solutions:
                            return None

                        # Prefer solutions with more 2-digit numbers, then fewer 1-digit numbers, then more numbers
                        def score(sol):
                            two_digit = sum(1 for n in sol if 10 <= n <= 49)
                            one_digit = sum(1 for n in sol if 1 <= n <= 9)
                            return (two_digit, -one_digit, len(sol))

                        best = max(solutions, key=score)
                        return best

                    def normalize_from_last(value):
                        if pd.isna(value):
                            return value
                        # Force to string while preserving integer-looking floats
                        if isinstance(value, float) and value.is_integer():
                            s = str(int(value))
                        else:
                            s = str(value)
                        # If we already have multiple numeric tokens, normalize them
                        tokens = re.findall(r"\d+", s)
                        if len(tokens) >= 2:
                            return " / ".join(tokens)
                        if "__BR__" in s:
                            parts = [p.strip() for p in s.split("__BR__") if p.strip()]
                            return " / ".join(parts)
                        # If we lost line breaks and got concatenated digits, try to split into 2-3 valid numbers
                        if re.fullmatch(r"\d{3,8}", s):
                            parts = split_compact_numbers(s)
                            if parts:
                                return " / ".join(str(n) for n in parts)
                        # Clean trailing .0 from numeric strings
                        s = re.sub(r"(?<=\d)\.0$", "", s)
                        return s

                    df['From Last'] = df['From Last'].apply(normalize_from_last)

                if latest_draw is not None and 'Draw' in df.columns:
                    df_temp = df.copy()
                    df_temp['Draw'] = pd.to_numeric(df_temp['Draw'], errors='coerce')
                    
                    # Check if this page has any old data
                    min_draw_in_page = df_temp['Draw'].min()
                    if pd.notna(min_draw_in_page) and min_draw_in_page <= latest_draw:
                        # This page contains old data, filter it
                        new_rows = df_temp[df_temp['Draw'] > latest_draw]
                        if len(new_rows) > 0:
                            return (page, new_rows, None)
                        else:
                            return (page, None, "All data already exists")
                    else:
                        return (page, df, None)
                else:
                    return (page, df, None)
        
        return (page, None, "No relevant table found")
        
    except Exception as e:
        return (page, None, f"Error: {str(e)}")

def scrape_toto_history():
    """
    Scrapes TOTO lottery history with incremental update logic:
    1. Checks for existing file with today's date
    2. If exists, only downloads new data not in the file
    3. If creating new file, backs up old files first
    """
    base_url = "https://en.lottolyzer.com/history/singapore/toto/page/{}/per-page/50/summary-view"
    
    # Generate filename with date code (e.g., ToTo-16_Jan_2026.csv)
    current_date = datetime.now()
    date_code = current_date.strftime("%d_%b_%Y")  # Format: 16_Jan_2026
    output_filename = f"ToTo-{date_code}.csv"
    output_file = os.path.join(os.getcwd(), output_filename)
    
    # Step 1: Check if today's file exists
    existing_data = None
    latest_draw = None
    
    if os.path.exists(output_file):
        print(f"Found existing file: {output_filename}")
        latest_draw = get_latest_draw_from_file(output_file)
        if latest_draw:
            print(f"Latest draw in file: {latest_draw}")
            print("Will only download newer data...")
            # Load existing data to merge later
            existing_data = pd.read_csv(output_file)
        else:
            print("Could not determine latest draw. Will scrape all data.")
    else:
        print(f"No existing file found for today. Will create: {output_filename}")
        # Step 3: Backup old files before creating new one
        backup_old_files(output_filename, os.getcwd())
    
    
    print("\nStarting multi-threaded scraping with 8 threads...")
    print_lock = threading.Lock()
    
    # Multi-threaded scraping
    all_data = []
    max_pages = 100  # Safety limit
    num_threads = 8
    
    # First, estimate how many pages we need to check
    # Start with batches of pages
    current_batch_start = 1
    batch_size = num_threads * 2  # Process 16 pages at a time
    found_end = False
    
    while not found_end and current_batch_start <= max_pages:
        # Prepare batch of pages to scrape
        pages_to_scrape = range(current_batch_start, min(current_batch_start + batch_size, max_pages + 1))
        
        # Scrape pages in parallel
        batch_results = {}
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all pages in this batch
            future_to_page = {
                executor.submit(scrape_single_page, page, base_url, latest_draw): page 
                for page in pages_to_scrape
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page, df, error = future.result()
                    batch_results[page] = (df, error)
                    
                    with print_lock:
                        if df is not None:
                            print(f"✓ Page {page:3d} - Found {len(df)} rows", end='   \r')
                        elif error and "already exists" in error:
                            print(f"⊗ Page {page:3d} - {error}", end='   \r')
                        elif error:
                            print(f"✗ Page {page:3d} - {error}", end='   \r')
                            
                except Exception as e:
                    with print_lock:
                        print(f"✗ Page {page_num:3d} - Exception: {e}", end='   \r')
                    batch_results[page_num] = (None, str(e))
        
        # Process results in order
        for page in sorted(batch_results.keys()):
            df, error = batch_results[page]
            
            if df is not None and len(df) > 0:
                all_data.append(df)
            elif error:
                # Check if we've hit the end
                if "already exists" in error or "No tables found" in error or "No relevant table" in error:
                    found_end = True
                    with print_lock:
                        print(f"\nStopping: Page {page} - {error}")
                    break
        
        # Move to next batch
        current_batch_start += batch_size
        
        # If we didn't find any data in this batch, stop
        if not any(df is not None for df, _ in batch_results.values()):
            found_end = True
            print("\nNo more data found. Stopping.")
    
    print(f"\nScraping complete. Processed up to page {current_batch_start - 1}")

    
    # Process and save data
    if all_data or existing_data is not None:
        if all_data:
            new_df = pd.concat(all_data, ignore_index=True)
            print(f"\nScraped {len(new_df)} new rows from website.")
            
            # Clean the new data
            cleaned_new_df = clean_scraped_data(new_df)
            
            # Merge with existing data if applicable
            if existing_data is not None:
                print(f"Merging with {len(existing_data)} existing rows...")
                # Combine and remove duplicates based on Draw number
                combined_df = pd.concat([existing_data, cleaned_new_df], ignore_index=True)
                
                # Remove duplicates, keeping the newest version
                if 'Draw' in combined_df.columns:
                    combined_df['Draw'] = pd.to_numeric(combined_df['Draw'], errors='coerce')
                    combined_df = combined_df.drop_duplicates(subset=['Draw'], keep='last')
                    combined_df = combined_df.sort_values('Draw', ascending=False).reset_index(drop=True)
                
                final_df = combined_df
                print(f"Final dataset has {len(final_df)} total rows.")
            else:
                final_df = cleaned_new_df
        else:
            # No new data scraped, but we have existing data
            print("\nNo new data to scrape. Existing file is up to date.")
            final_df = existing_data
        
        # Save final data
        final_df.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")
        
        # Display first few rows to verify
        print("\nFirst 5 rows of data:")
        print(final_df.head())
    else:
        print("\nNo data collected and no existing data found.")

if __name__ == "__main__":
    scrape_toto_history()
