import pandas as pd
import sys
import os
from datetime import datetime
import html

# Default input file (local workspace) or user path
INPUT_FILE = f"4D_results_{datetime.now().strftime('%d_%b_%Y')}.csv"
date_str = datetime.now().strftime("%d_%b_%Y")
OUTPUT_REPORT = f"Analysis_4D_{date_str}.md"
OUTPUT_REPORT_HTML = f"Analysis_4D_{date_str}.html"

def analyze_stats(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records.")
    
    # Pre-processing
    df['Draw Date'] = pd.to_datetime(df['Draw Date'])
    df['Number'] = df['Number'].astype(str).str.zfill(4)
    
    # Create a full list of 0000-9999
    all_numbers = [f"{i:04d}" for i in range(10000)]
    
    report_lines = []
    report_lines.append("# Singapore 4D Statistical Analysis")
    report_lines.append(f"**Total Records Analyzed**: {len(df)}")
    report_lines.append(f"**Data Source**: `{file_path}`")
    report_lines.append("")

    # --- Pre-calculate Stat Lookup Tables ---
    
    # 1. Frequency per number
    freq_series = df['Number'].value_counts().reindex(all_numbers, fill_value=0)
    
    # 2. Last Draw Info per number
    # We want the row with the max date for each number.
    # Sort by date desc so first item per group is the latest
    df_sorted = df.sort_values('Draw Date', ascending=False)
    last_draw_df = df_sorted.drop_duplicates('Number', keep='first').set_index('Number')
    
    # Reindex to include numbers that never won (fill NaT/NaN)
    last_draw_df = last_draw_df.reindex(all_numbers)
    
    # Helper to get last draw info
    def get_last_info(num):
        if num not in last_draw_df.index:
            return "Never", "N/A"
        row = last_draw_df.loc[num]
        if pd.isna(row['Draw Date']):
            return "Never", "N/A"
        return row['Draw Date'].strftime('%Y-%m-%d'), row['Prize Group']

    # --- A) Least Drawn Numbers ---
    report_lines.append("## A) Least Drawn Numbers")
    sorted_freq = freq_series.sort_values(ascending=True)
    
    for top_n in [10, 20, 50]:
        report_lines.append(f"### Top {top_n} Least Drawn")
        subset = sorted_freq.head(top_n)
        
        report_lines.append("| Number | Frequency | Last Draw Date | Last Prize Group |")
        report_lines.append("|---|---|---|---|")
        for num, count in subset.items():
            last_date, last_prize = get_last_info(num)
            report_lines.append(f"| `{num}` | {int(count)} | {last_date} | {last_prize} |")
        report_lines.append("")

    # --- B) Numbers with Oldest Last Draw Date ---
    report_lines.append("## B) Numbers with Oldest Last Draw Date")
    report_lines.append("> Includes frequency count.")
    
    # Create stats DF
    stats_df = pd.DataFrame({
        'Frequency': freq_series,
        'Last Draw': last_draw_df['Draw Date'],
        'Last Prize': last_draw_df['Prize Group']
    })
    
    # Sort: NaT (Never) first, then Oldest dates
    stats_df = stats_df.sort_values(by='Last Draw', ascending=True, na_position='first')
    
    # Format date for display
    stats_df['Formatted Date'] = stats_df['Last Draw'].dt.strftime('%Y-%m-%d').fillna('Never')
    stats_df['Last Prize'] = stats_df['Last Prize'].fillna('N/A')
    
    for top_n in [10, 20, 50]:
        report_lines.append(f"### Top {top_n} Oldest Last Draw")
        subset = stats_df.head(top_n)
        report_lines.append("| Number | Last Draw Date | Total Occurrence | Last Prize Group |")
        report_lines.append("|---|---|---|---|")
        for num, row in subset.iterrows():
            report_lines.append(f"| `{num}` | {row['Formatted Date']} | {int(row['Frequency'])} | {row['Last Prize']} |")
        report_lines.append("")

    # --- C) Most Frequent Combinations ---
    report_lines.append("## C) Most Frequent Combinations")
    report_lines.append("> Expanded to Top 10/20/50. Includes last occurrence info.")

    # Generate combination key
    def get_signature(num_str):
        return "".join(sorted(num_str))
        
    df['Combination'] = df['Number'].apply(get_signature)
    
    # Count frequency of combinations
    combo_counts = df['Combination'].value_counts()
    
    # Find last draw for each combination
    # Sort by Date Desc, drop duplicates by Combo -> gives latest row for that combo
    last_combo_draw = df.sort_values('Draw Date', ascending=False).drop_duplicates('Combination', keep='first').set_index('Combination')

    # Build last 5 draw history (date + prize group) per combination
    combo_recent = df.sort_values('Draw Date', ascending=False).groupby('Combination').head(5).copy()
    combo_recent['Draw Date Str'] = combo_recent['Draw Date'].dt.strftime('%Y-%m-%d')
    combo_recent['Prize Group'] = combo_recent['Prize Group'].fillna('N/A')
    last5_combo = combo_recent.groupby('Combination')[['Draw Date Str', 'Prize Group']].apply(
        lambda g: "<br>".join(
            f"{d} ({p})" for d, p in zip(g['Draw Date Str'], g['Prize Group'])
        )
    )

    for top_n in [10, 20, 50]:
        report_lines.append(f"### Top {top_n} Combinations")
        subset = combo_counts.head(top_n)
        
        report_lines.append("| Combination | Frequency | Last 5 Draws (Date, Prize Group) | Example # |")
        report_lines.append("|---|---|---|---|")
        
        for combo, count in subset.items():
            # Get last draw details
            if combo in last_combo_draw.index:
                row = last_combo_draw.loc[combo]
                example = row['Number'] # The specific number that triggered the last draw
            else:
                example = "N/A"
            last5 = last5_combo.get(combo, "N/A")
            
            report_lines.append(f"| `{combo}` | {count} | {last5} | `{example}` |")
        report_lines.append("")

    # Write markdown report
    with open(OUTPUT_REPORT, "w", encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    # Convert markdown to HTML (minimal for this report format)
    def format_inline(text):
        placeholder = "__BR__PLACEHOLDER__"
        text = text.replace("<br>", placeholder)
        parts = text.split("`")
        for i in range(len(parts)):
            parts[i] = html.escape(parts[i])
            if i % 2 == 1:
                parts[i] = f"<code>{parts[i]}</code>"
        rendered = "".join(parts)
        rendered = rendered.replace("**", "<strong>", 1).replace("**", "</strong>", 1) if "**" in rendered else rendered
        return rendered.replace(placeholder, "<br>")

    html_lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<title>Singapore 4D Statistical Analysis</title>",
        "<style>",
        "body{font-family:Arial, sans-serif; margin:24px; color:#111;}",
        "h1,h2,h3{margin-top:24px;}",
        "table{border-collapse:collapse; margin:12px 0; width:100%;}",
        "th,td{border:1px solid #ccc; padding:6px 8px; text-align:left; vertical-align:top;}",
        "code{background:#f6f6f6; padding:1px 4px; border-radius:3px;}",
        "blockquote{border-left:3px solid #ddd; margin:8px 0; padding:4px 10px; color:#555;}",
        "</style>",
        "</head>",
        "<body>"
    ]

    i = 0
    while i < len(report_lines):
        line = report_lines[i]
        if line.startswith("|") and i + 1 < len(report_lines) and report_lines[i + 1].startswith("|---"):
            header = [c.strip() for c in line.strip("|").split("|")]
            i += 2  # skip header separator
            rows = []
            while i < len(report_lines) and report_lines[i].startswith("|"):
                cells = [c.strip() for c in report_lines[i].strip("|").split("|")]
                rows.append(cells)
                i += 1
            html_lines.append("<table>")
            html_lines.append("<thead><tr>" + "".join(f"<th>{format_inline(c)}</th>" for c in header) + "</tr></thead>")
            html_lines.append("<tbody>")
            for row in rows:
                html_lines.append("<tr>" + "".join(f"<td>{format_inline(c)}</td>" for c in row) + "</tr>")
            html_lines.append("</tbody></table>")
            continue
        if line.startswith("### "):
            html_lines.append(f"<h3>{format_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{format_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            html_lines.append(f"<h1>{format_inline(line[2:])}</h1>")
        elif line.startswith("> "):
            html_lines.append(f"<blockquote>{format_inline(line[2:])}</blockquote>")
        elif line.strip() == "":
            html_lines.append("<br>")
        else:
            html_lines.append(f"<p>{format_inline(line)}</p>")
        i += 1

    html_lines.append("</body></html>")

    with open(OUTPUT_REPORT_HTML, "w", encoding='utf-8') as f:
        f.write("\n".join(html_lines))
        
    print(f"Analysis complete. Report saved to {OUTPUT_REPORT} and {OUTPUT_REPORT_HTML}")

if __name__ == "__main__":
    # Check if a specific file path was passed or local default
    # For the user, they might run "python Analyze_4D.py" which looks for "4d_results.csv" in CWD
    target_file = INPUT_FILE
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    
    analyze_stats(target_file)
