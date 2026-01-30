"""
Singapore Pools 4D Lottery Simulation - Advanced Dynamic Version
Simulates static and dynamic betting strategies with performance optimization.

Features:
- Dynamic Strategy Engine (Martingale, Paroli, etc.)
- Parallel Multiprocessing
- Detailed 'Best Run' Tracking
- Enhanced Statistical Analysis
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import time
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Prize structure constants (per $1 bet)
PRIZES_BIG = {
    '1st': 2000,
    '2nd': 1000,
    '3rd': 490,
    'starter': 250,
    'consolation': 60
}

PRIZES_SMALL = {
    '1st': 3000,
    '2nd': 2000,
    '3rd': 800
}

def generate_winning_numbers():
    """Generate 23 unique winning numbers for a 4D draw."""
    # Optimization: Use random.sample directly on range which is faster
    winning_numbers = random.sample(range(10000), 23)
    
    # Store as hash set for O(1) lookups during win check
    # But we also need category mapping
    return {
        '1st': {winning_numbers[0]},
        '2nd': {winning_numbers[1]},
        '3rd': {winning_numbers[2]},
        'starter': set(winning_numbers[3:13]),
        'consolation': set(winning_numbers[13:23]),
        # Helper set containing ALL winning numbers for fast initial check
        'all': set(winning_numbers)
    }

def calculate_winnings(tickets, winning_numbers, big_bet_amount, small_bet_amount):
    """
    Calculate total winnings for a set of tickets.
    Returns: (total_winnings, details_dict)
    """
    total_winnings = 0
    details = {
        'Win_1st': 0, 'Win_2nd': 0, 'Win_3rd': 0, 
        'Win_Starter': 0, 'Win_Consolation': 0
    }
    
    # Optimization: Pre-check if we have any wins at all
    all_winning_nums = winning_numbers['all']
    
    for ticket in tickets:
        if ticket not in all_winning_nums:
            continue
            
        is_winner = False
        # Check specific categories in order of value
        if big_bet_amount > 0:
            if ticket in winning_numbers['1st']: 
                total_winnings += 2000 * big_bet_amount
                details['Win_1st'] += 1
                is_winner = True
            elif ticket in winning_numbers['2nd']: 
                total_winnings += 1000 * big_bet_amount
                details['Win_2nd'] += 1
                is_winner = True
            elif ticket in winning_numbers['3rd']: 
                total_winnings += 490 * big_bet_amount
                details['Win_3rd'] += 1
                is_winner = True
            elif ticket in winning_numbers['starter']: 
                total_winnings += 250 * big_bet_amount
                details['Win_Starter'] += 1
                is_winner = True
            elif ticket in winning_numbers['consolation']: 
                total_winnings += 60 * big_bet_amount
                details['Win_Consolation'] += 1
                is_winner = True
            
        if small_bet_amount > 0:
            # Note: A ticket can win both Big and Small if placed on same number
            # We don't double count the "Win_X" statistic for same number, 
            # but usually these are separate tickets in this sim model or same ticket logic.
            # For simplicity, we increment counters based on the Match, not the Bet Type pay.
            # If we already matched above, we just add money.
            
            if ticket in winning_numbers['1st']: 
                total_winnings += 3000 * small_bet_amount
                if not is_winner: details['Win_1st'] += 1
            elif ticket in winning_numbers['2nd']: 
                total_winnings += 2000 * small_bet_amount
                if not is_winner: details['Win_2nd'] += 1
            elif ticket in winning_numbers['3rd']: 
                total_winnings += 800 * small_bet_amount
                if not is_winner: details['Win_3rd'] += 1

    return total_winnings, details

# -----------------------------------------------------------------------------
# DYNAMIC STRATEGY ENGINE
# -----------------------------------------------------------------------------

class SimulationState:
    def __init__(self, initial_balance=0):
        self.balance = initial_balance
        self.history = []  # List of previous profit/loss
        self.streak = 0    # Positive for win streak, negative for loss streak
        self.current_draw = 0
        self.bankroll = 10000 + initial_balance # Simulation starting bankroll
        self.peak_balance = initial_balance # For Ratchet/Stop-loss
        self.virtual_streak = 0 # For Sniper (paper trading)

def get_bet_parameters_dynamic(strategy_name, state):
    """
    Decides bet parameters based on strategy logic and current state.
    Returns: (num_tickets, big_bet, small_bet)
    """
    
    # Update Peak Balance
    if state.balance > state.peak_balance:
        state.peak_balance = state.balance

    # --- STATIC STRATEGIES (Baseline) ---
    if strategy_name == 'Static: $1 Big/$1 Small (1000 tix)': return 1000, 1, 1
    if strategy_name == 'Static: $10 Small (100 tix)': return 100, 0, 10

    # --- DYNAMIC STRATEGIES (New Variants) ---
    
    # 1. MARTINGALE VARIATIONS
    
    # Baseline: Small, 50 Tix
    if strategy_name == 'Dynamic: Martingale (Small, 50 tix)':
        base_bet = 10
        multiplier = 1
        if state.streak < 0:
            loss_count = abs(state.streak)
            multiplier = 2 ** min(loss_count, 6)
        return 50, 0, base_bet * multiplier

    # Variant A: Big Bet (High Frequency Wins)
    if strategy_name == 'Dynamic: Martingale (Big, 50 tix)':
        base_bet = 10
        multiplier = 1
        if state.streak < 0:
            loss_count = abs(state.streak)
            multiplier = 2 ** min(loss_count, 6)
        return 50, base_bet * multiplier, 0

    # Variant B: High Volume (200 Tix)
    if strategy_name == 'Dynamic: Martingale (Small, 200 tix)':
        base_bet = 10
        multiplier = 1
        if state.streak < 0:
            # Note: With 200 tix, cost is 4x. We keep base bet $10/ticket.
            loss_count = abs(state.streak)
            multiplier = 2 ** min(loss_count, 6)
        return 200, 0, base_bet * multiplier

    # 4. SNIPER VARIATIONS
    
    # Baseline: Small, 50 Tix, Wait 10
    if strategy_name == 'Smart: Sniper (Small, 50 tix)':
        if state.virtual_streak > -8: return 1, 0, 0 # Paper trade
        else: return 50, 0, 50 # Attack

    # Variant C: Big Bet + Med Volume
    if strategy_name == 'Smart: Sniper (Big, 100 tix)':
        # Big bets win more often, so maybe wait for longer streak? 
        # Or just keep 8.
        if state.virtual_streak > -8: return 1, 0, 0
        else: return 100, 50, 0 # Attack with Big Bets

    # 5. HYBRID PAROLI VARIATIONS
    if strategy_name == 'Smart: Paroli (Big, 50 tix)':
        base_bet = 10
        if state.streak > 0:
            multiplier = 4 ** min(state.streak, 3) 
            return 50, base_bet * multiplier, 0
        else:
            return 50, base_bet, 0

    # --- STRATEGY 4.0 (COVERAGE / SWARM) ---
    
    # 6. COVERAGE MARTINGALE (Swarm Attack)
    # Logic: Start 50 tix. Lose? Buy 100. Lose? Buy 200.
    # Cost doubles, but hit rate doubles.
    if strategy_name == 'Coverage: Martingale (Small)':
        base_tix = 50
        multiplier = 1
        if state.streak < 0:
            loss_count = abs(state.streak)
            multiplier = 2 ** min(loss_count, 7) # Cap at 128x (6400 tickets)
            
        tix_count = base_tix * multiplier
        return tix_count, 0, 10
        
    # 7. COVERAGE PAROLI (Swarm Rider)
    # Logic: Start 50 tix. Win? Buy 200.
    if strategy_name == 'Coverage: Paroli (Small)':
        base_tix = 50
        multiplier = 1
        if state.streak > 0:
            win_count = min(state.streak, 4)
            multiplier = 2 ** win_count
            
        tix_count = base_tix * multiplier
        return tix_count, 0, 10

    # --- STRATEGY 5.0 (ADAPTIVE SWARM) ---
    if strategy_name == 'Dynamic: Adaptive Swarm':
        # DEFENSE (Loss Streak) -> Swarm Big
        if state.streak < 0:
            loss_count = abs(state.streak)
            # Default
            num_tix = 100
            if loss_count >= 4: num_tix = 300
            if loss_count >= 7: num_tix = 600
            
            # Bet BIG (Easier to break loss streak) but small size ($1)
            return num_tix, 1, 0
            
        # OFFENSE (Win Streak) -> Sniper Small
        elif state.streak > 0:
            # We are hot. Go for ROI.
            # 50 Tickets (Focus)
            # Scale Bet Size (Paroli)
            bet_amt = 10 # Base
            if state.streak >= 2: bet_amt = 50
            if state.streak >= 3: bet_amt = 200
            
            # Bet SMALL (High Payout)
            return 50, 0, bet_amt
            
        # NEUTRAL (Start)
        else:
            return 50, 0, 10

    # --- STRATEGY SET 6 (THE FINAL EXPANSION) ---

    # 4. STATIC: HIGH COVERAGE (Big) - The Control
    # Always 200 tickets. No changes.
    if strategy_name == 'Static: High Coverage (200 tix)':
        return 200, 1, 0  # 200 tickets, $1 Big

    # 5. DYNAMIC: LINEAR RECOVERY (Big) - The Soft Defender
    # Add 50 tickets on loss. Reset on win.
    if strategy_name == 'Dynamic: Linear Recovery (Big)':
        base_tix = 50
        step_size = 50
        if state.streak < 0:
            loss_count = abs(state.streak)
            # Linear increase: 50 + (Losses * 50)
            # Cap at 1000 tickets to prevent total memory crash
            add_on = min(loss_count, 20) * step_size 
            return base_tix + add_on, 1, 0
        return base_tix, 1, 0

    # 6. DYNAMIC: INFINITE PAROLI (Small) - The Moonshot
    # Double bet size on win FOREVER.
    if strategy_name == 'Dynamic: Infinite Paroli (Small)':
        base_bet = 10
        multiplier = 1
        if state.streak > 0:
            # NO CAP. Let it ride.
            # 2^10 = 1024x. 2^20 = 1,000,000x.
            # Warning: Float overflow protection.
            win_count = state.streak
            if win_count > 20: win_count = 20 # Cap at becoming Billionaire
            multiplier = 2 ** win_count
            
        return 50, 0, base_bet * multiplier

    # Fallback
    return 50, 0, 10

    return 100, 0, 10

def simulate_single_full_run(strategy_name, num_draws=1000):
    state = SimulationState()
    cumulative_profit = []
    total_profit = 0
    detailed_history = []
    
    for draw_num in range(1, num_draws + 1):
        state.current_draw = draw_num
        
        # 1. Decide Bet
        num_tickets, big_bet, small_bet = get_bet_parameters_dynamic(strategy_name, state)
        cost = num_tickets * (big_bet + small_bet)
        
        # 2. Play Game
        winning_nums = generate_winning_numbers()
        player_tickets = [random.randint(0, 9999) for _ in range(num_tickets)]
        
        # 3. Calc Result (Real & Virtual)
        # We need to know if we WOULD have won even if we bet $0 (for Sniper)
        # So we always generate tickets and check them.
        winnings, win_details = calculate_winnings(player_tickets, winning_nums, big_bet, small_bet)
        
        # Virtual Check: Did we get ANY match? (For Sniper virtual streak)
        # We assume "Virtual Win" if any ticket matched any prize
        virtual_win = (win_details['Win_1st'] + win_details['Win_2nd'] + win_details['Win_3rd'] + 
                      win_details['Win_Starter'] + win_details['Win_Consolation']) > 0
        
        if virtual_win:
            if state.virtual_streak < 0: state.virtual_streak = 1
            else: state.virtual_streak += 1
        else:
            if state.virtual_streak > 0: state.virtual_streak = -1
            else: state.virtual_streak -= 1
        
        # Real Profit
        profit = winnings - cost
        total_profit += profit
        
        # 4. Update State
        state.balance = total_profit
        state.history.append(profit)
        cumulative_profit.append(total_profit)
        
        if profit > 0:
            if state.streak < 0: state.streak = 1
            else: state.streak += 1
        else:
            if state.streak > 0: state.streak = -1
            else: state.streak -= 1
            
        # 5. Log Significant Events
        row = {
            'Draw': draw_num,
            'Strategy': strategy_name,
            'Tickets': num_tickets,
            'Big_Bet': big_bet,
            'Small_Bet': small_bet,
            'Cost': cost,
            'Winnings': winnings,
            'Profit_Loss': profit,
            'Cumulative_Balance': total_profit,
            'Streak_State': state.streak,
            # Add Win Details
            'Win_1st_Cnt': win_details['Win_1st'],
            'Win_2nd_Cnt': win_details['Win_2nd'],
            'Win_3rd_Cnt': win_details['Win_3rd'],
            'Win_Starter_Cnt': win_details['Win_Starter'],
            'Win_Consolation_Cnt': win_details['Win_Consolation']
        }
        detailed_history.append(row)
            
    return cumulative_profit, detailed_history

# -----------------------------------------------------------------------------
# PARALLEL EXECUTION ENGINE
# -----------------------------------------------------------------------------

def worker_simulation(params):
    """Worker function for parallel execution"""
    strategy_name, num_draws, sim_id = params
    
    # Run simulation
    cum_profit, history = simulate_single_full_run(strategy_name, num_draws)
    
    final_pl = cum_profit[-1]
    best_point = max(cum_profit)
    worst_point = min(cum_profit)
    
    # Return summary stats + full history ONLY if it's exceptionally good 
    # (Checking against a threshold helps reduced IPC overhead, but we'll return all
    # and filter in main for simplicity unless it's too huge)
    return {
        'sim_id': sim_id,
        'strategy': strategy_name,
        'final_pl': final_pl,
        'best_point': best_point,
        'worst_point': worst_point,
        'cum_profit': cum_profit, # Needed for average plot
        'history': history        # Needed for best run log
    }

def run_advanced_simulation(num_meta_sims=50, num_draws=1000):
    strategies = [
        'Dynamic: Adaptive Swarm',
        'Coverage: Paroli (Small)',
        'Dynamic: Martingale (Small, 50 tix)',
        'Static: High Coverage (200 tix)',
        'Dynamic: Linear Recovery (Big)',
        'Dynamic: Infinite Paroli (Small)'
    ]
    
    # Store aggregated results
    agg_results = defaultdict(lambda: {'final_pl': [], 'all_runs': []})
    
    # Track absolute best run across ALL strategies
    global_best_run = {'profit': -float('inf'), 'history': None, 'strategy': None, 'id': None}
    
    cpu_count = multiprocessing.cpu_count()
    print(f"\n[SYSTEM] Detected {cpu_count} CPU cores. Launching parallel engine...")
    
    total_tasks = len(strategies) * num_meta_sims
    completed = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = []
        
        # Submit all tasks
        for strategy in strategies:
            for i in range(num_meta_sims):
                futures.append(executor.submit(worker_simulation, (strategy, num_draws, i)))
                
        print(f"[SYSTEM] submitted {total_tasks} simulation tasks...")
        
        # Process results as they finish
        for future in as_completed(futures):
            res = future.result()
            strat = res['strategy']
            
            # 1. Store Data
            agg_results[strat]['final_pl'].append(res['final_pl'])
            agg_results[strat]['all_runs'].append(res['cum_profit'])
            
            # 2. Check for Best Run
            if res['final_pl'] > global_best_run['profit']:
                global_best_run = {
                    'profit': res['final_pl'],
                    'history': res['history'],
                    'strategy': strat,
                    'id': res['sim_id']
                }
            
            # 3. Progress Update
            completed += 1
            if completed % (max(1, total_tasks // 20)) == 0 or completed == total_tasks:
                elapsed = time.time() - start_time
                print(f"[PROGRESS] {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%) "
                      f"- Best so far: ${global_best_run['profit']:,.0f} ({global_best_run['strategy']})")

    # Final Average Calculation
    print("\n=== PERFORMANCE REPORT (100 Runs) ===")
    winner_strat = None
    winner_wins = -1
    
    for strat in strategies:
        # Calculate stats
        final_pls = agg_results[strat]['final_pl']
        avg_final = np.mean(final_pls)
        
        # Count Profitable Runs (> 0)
        profitable_runs = sum(1 for x in final_pls if x > 0)
        win_rate = (profitable_runs / num_meta_sims) * 100
        
        print(f"Strategy: {strat:40} | Avg: ${avg_final:,.0f} | Win Rate: {win_rate:.1f}% ({profitable_runs}/{num_meta_sims})")
        
        if profitable_runs > winner_wins:
            winner_wins = profitable_runs
            winner_strat = strat
            
    print(f"\n[WINNER] (Most Profitable Runs): {winner_strat} ({winner_wins} wins)")
        
    return agg_results, global_best_run

# -----------------------------------------------------------------------------
# VISUALIZATION & OUTPUT
# -----------------------------------------------------------------------------

def save_best_run_log(best_run_data, filename):
    if not best_run_data or not best_run_data['history']:
        return
        
    history = best_run_data['history']
    keys = history[0].keys()
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    print(f"\n[OUTPUT] Detailed history of best run saved to: {filename}")

def plot_advanced_results(agg_results, filename):
    strategies = list(agg_results.keys())
    
    # Grid Layout: 3 rows x 2 columns (for 6 strategies)
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, strat in enumerate(strategies):
        if i >= len(axes): break
        ax = axes[i]
        
        all_runs = agg_results[strat]['all_runs']
        
        # 1. Plot spaghetti lines (Individual runs)
        # Convert to numpy for faster plotting if huge, but list loop is fine for 50
        for run in all_runs:
            ax.plot(run, color='blue', alpha=0.15, linewidth=1)
            
        # 2. Plot Average Line
        # Calculate element-wise mean
        avg_curve = np.mean(np.array(all_runs), axis=0)
        ax.plot(avg_curve, color='red', linewidth=2.5, label='Average P/L')
        
        ax.set_title(strat, fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Balance')
        ax.set_xlabel('Draw Number')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Calc Mean Final Profit
        mean_final = avg_curve[-1]
        ax.text(0.95, 0.05, f"Avg Final:\n${mean_final:,.0f}", 
                transform=ax.transAxes, ha='right', va='bottom', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"[OUTPUT] Analysis charts saved to: {filename}")

def get_timestamped_filename(base):
    return f"{base}_{datetime.now().strftime('%d%b%Y_%H%M')}"

if __name__ == "__main__":
    multiprocessing.freeze_support() # Windows support
    
    print("="*60)
    print("SINGAPORE POOLS 4D - DYNAMIC STRATEGY ENGINE")
    print("Using Parallel Processing for High-Performance Simulation")
    print("="*60)
    
    # Configuration
    SIMS = 100
    DRAWS = 1000
    
    # Run
    results, best_run = run_advanced_simulation(SIMS, DRAWS)
    
    # Generate Filenames
    base_name = get_timestamped_filename("4d_royal_rumble")
    plot_file = f"{base_name}.png"
    log_file = f"{base_name}_BEST_RUN_HISTORY.csv"
    
    # Save Outputs
    plot_advanced_results(results, plot_file)
    save_best_run_log(best_run, log_file)
    
    # Summary
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print(f"Top Performing Strategy Run: {best_run['strategy']}")
    print(f"Final Profit: ${best_run['profit']:,.2f}")
    print("="*60)
