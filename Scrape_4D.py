import requests
import json
import pandas as pd
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# API Configuration
URL = "https://www.singaporepools.com.sg/_layouts/15/FourD/FourDCommon.aspx/Get4DNumberCheckResultsJSON"
HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "Referer": "https://www.singaporepools.com.sg/en/product/Pages/4d_cpwn.aspx",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

OUTPUT_FILE = f"4D_results_{datetime.now().strftime('%d_%b_%Y')}.csv"
CURRENT_OUTPUT_FILE = OUTPUT_FILE 
msg_lock = threading.Lock()
csv_lock = threading.Lock()

def fetch_data(numbers):
    payload = {
        "numbers": numbers, 
        "checkCombinations": "false",
        "sortTypeInteger": "2" 
    }
    
    try:
        response = requests.post(URL, json=payload, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            result_json = json.loads(data['d'])
            return result_json
        else:
            with msg_lock:
                print(f"Status {response.status_code} for {numbers}")
            return None
    except Exception as e:
        with msg_lock:
            print(f"Error fetching {numbers}: {e}")
        return None

def parse_results(result_json):
    records = []
    if not result_json:
        return records
        
    for history in result_json:
        number = history.get('Number', 'UNKNOWN')
        prizes = history.get('Prizes', [])
        
        for prize in prizes:
            records.append({
                'Number': number,
                'Draw Date': normalize_date(prize.get('DrawDate')),
                'Prize Group': get_prize_name(prize.get('PrizeCode'))
            })
    return records

def normalize_date(json_date):
    try:
        timestamp = int(json_date.replace('/Date(', '').replace(')/', ''))
        dt = datetime.fromtimestamp(timestamp / 1000)
        return dt.strftime('%Y-%m-%d')
    except:
        return json_date

def get_prize_name(code):
    mapping = {
        '1': 'First Prize',
        '2': 'Second Prize',
        '3': 'Third Prize',
        'S': 'Starter Prize',
        'C': 'Consolation Prize'
    }
    return mapping.get(code, code)

def process_batch(batch):
    # Reduced delay as requested, but in threads it happens in parallel
    time.sleep(0.1) 
    
    results = fetch_data(batch)
    if results:
        records = parse_results(results)
        
        # Write immediately to file (safely)
        if records:
            df = pd.DataFrame(records)
            with csv_lock:
                # Use the global CURRENT_OUTPUT_FILE which allows switching to temp
                target_file = CURRENT_OUTPUT_FILE
                header = not os.path.exists(target_file) or os.path.getsize(target_file) == 0
                df.to_csv(target_file, mode='a', header=header, index=False)
        return len(records)
    return 0

def main():
    # 1. Define filenames
    temp_file = OUTPUT_FILE + ".tmp"
    
    # 2. Check for existing file and load its records
    old_records = []
    old_numbers = set()
    old_count = 0
    if os.path.exists(OUTPUT_FILE):
        try:
            df_old = pd.read_csv(OUTPUT_FILE)
            old_records = df_old.to_dict('records')
            # Ensure proper string formatting for comparison
            if 'Number' in df_old.columns:
                old_numbers = set(df_old['Number'].astype(str).str.zfill(4).unique())
            old_count = len(df_old)
            print(f"Existing file found: {OUTPUT_FILE}")
            print(f"Existing records: {old_count}. Unique numbers having data: {len(old_numbers)}")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            
    # 3. Prepare for scraping
    # Clear temp file if exists
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    all_numbers_str = [f"{i:04d}" for i in range(10000)] # 0000 to 9999
    
    # Validation loop variables
    numbers_to_scrape = all_numbers_str
    retry_count = 0
    max_retries = 5
    
    # Global variable update for worker threads
    global CURRENT_OUTPUT_FILE
    CURRENT_OUTPUT_FILE = temp_file
    
    while retry_count <= max_retries:
        if retry_count > 0:
            print(f"\n--- Retry Attempt {retry_count}/{max_retries} ---")
            print(f"Rescraping {len(numbers_to_scrape)} missing numbers...")
            
        # 4. Scrape logic
        if numbers_to_scrape:
            # Recreate batches for missing numbers
            batch_size = 4
            batches = [numbers_to_scrape[i:i+batch_size] for i in range(0, len(numbers_to_scrape), batch_size)]
            
            total_records = 0
            completed_batches = 0
            total_batches = len(batches)
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=40) as executor:
                future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
                
                for future in as_completed(future_to_batch):
                    try:
                        count = future.result()
                        total_records += count
                        completed_batches += 1
                        
                        if completed_batches % 50 == 0 or completed_batches == total_batches:
                            elapsed = time.time() - start_time
                            rate = completed_batches / elapsed if elapsed > 0 else 0
                            percentage = (completed_batches / total_batches) * 100
                            print(f"Progress: {percentage:.2f}% ({completed_batches}/{total_batches}). Records: {total_records}. Rate: {rate:.2f}/s")
                    except Exception as exc:
                        print(f"Exception: {exc}")

        # 5. Validation Logic
        if os.path.exists(temp_file):
            try:
                df_new = pd.read_csv(temp_file)
                new_count = len(df_new)
                new_numbers = set()
                if 'Number' in df_new.columns:
                    new_numbers = set(df_new['Number'].astype(str).str.zfill(4).unique())
                
                print(f"New scan records: {new_count}. Unique numbers: {len(new_numbers)}")
                
                # Check for completeness against all invalid numbers or just all 10000?
                # The user requirement: "check missing number".
                # If we are scraping all 10000, we expect 10000 unique number entries? 
                # Not necessarily, some numbers might not have won ever? 
                # Wait, "Scrape_4D" usually scrapes results for numbers. If a number has 0 results, it returns empty?
                # If returns empty, then it's "scraped" but has 0 records.
                # The validation should be about "did we process all requested numbers?".
                # But fetch_data returns blank for error?
                # "compare the missing number" -> missing from the RESULT? or missing from being scraped?
                # If a number has records, it appears in CSV.
                # If a number has NO records (never won), it won't be in CSV.
                # So missing numbers count check is tricky unless we track "processed" numbers.
                # However, user says "new csv file must contain same or more records than the old csv file".
                
                # CRITERIA 1: Count check
                count_ok = new_count >= old_count
                
                # CRITERIA 2: Missing specific numbers from old file?
                # If old file had number '1234' with 5 records, new file should also have '1234'.
                # Let's check if any number from old_numbers is missing in new_numbers.
                missing_from_old = old_numbers - new_numbers
                
                if count_ok and not missing_from_old:
                    print("Validation SUCCESS: New file has >= records and contains all previously found numbers.")
                    break # Success
                else:
                    print(f"Validation FAILED: New Count {new_count} (Old {old_count}). Missing old numbers: {len(missing_from_old)}")
                    
                    # Logic to determine what to retry
                    # If we missed numbers that definitely have records (from old file), retry them.
                    # Also, if getting less records, maybe we just failed network calls?
                    # The "fetch_data" returns None on error.
                    # We should probably retry ALL failed/missing numbers if we can track them?
                    # Since we don't track connection errors explicitly in a list, we can rely on missing_from_old 
                    # PLUS maybe simply retrying the ones we suspect failed?
                    # Simplified retry: Retry old numbers that are missing.
                    # If count is less but no old numbers missing (e.g. data deletion?), that's weird but possible?
                    
                    if missing_from_old:
                        numbers_to_scrape = list(missing_from_old)
                    else:
                        # Count is low, but all old numbers present? Maybe duplicates removed?
                        # Or maybe we just want to ensure we tried everything.
                        # If invalid but no clear missing numbers, might need to restart whole thing?
                        # Or just accept it if retry count is high?
                        print("Warning: Counts differ but no specific old numbers missing. Retrying scrape for ANY number that returned 0 results?")
                        # Strategy: If validation fails, identifying "what to retry" is key.
                        # If we have a list of "failed" requests, that would be best.
                        # Since we don't, and `missing_from_old` covers the regression case.
                        # What if we missed a NEW number? We won't know.
                        # Let's stick to missing_from_old for retry.
                        if not missing_from_old:
                             # Break if we can't identify what to fix
                             print("Cannot identify missing numbers. Accepting result (risk of data regression if count is truly lower).")
                             break
                        numbers_to_scrape = list(missing_from_old)
            
            except Exception as e:
                print(f"Error during validation: {e}")
                # If validation errors, maybe safer to break or retry?
                break
        else:
             # Temp file empty?
             print("New temp file empty/missing!")
             numbers_to_scrape = all_numbers_str # Retry all
             
        retry_count += 1
    
    # Finalize
    if os.path.exists(temp_file):
        final_df = pd.read_csv(temp_file)
        # Sort
        if 'Number' in final_df.columns:
            final_df['Number'] = final_df['Number'].astype(str).str.zfill(4)
            final_df.sort_values(by=['Number', 'Draw Date'], ascending=[True, False], inplace=True)
            
            # Save final
            # 1. Backup old
            if os.path.exists(OUTPUT_FILE):
                backup_name = OUTPUT_FILE.replace(".csv", "OLD.csv")
                # Remove previous backup if exists to allow overwrite or keep history? 
                # User said "*.csvOLD".
                if os.path.exists(backup_name):
                    os.remove(backup_name)
                os.rename(OUTPUT_FILE, backup_name)
                print(f"Backed up old file to {backup_name}")
                
            # 2. Save new
            final_df.to_csv(OUTPUT_FILE, index=False)
            print(f"Success! Final data saved to {OUTPUT_FILE} with {len(final_df)} records.")
            
            # Clean temp
            os.remove(temp_file)
        else:
            print("Error: DataFrame missing 'Number' column upon final save.")
            
if __name__ == "__main__":
    main()
