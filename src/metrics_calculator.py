# This script contains all business logic for calculating agent performance.
# It is designed to be run after fetching the knowledge base and receiving data file paths.

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

class MetricsCalculator:
    def __init__(self, kb_path, start_date_str, end_date_str):
        self.kb_path = kb_path
        self.start_date = pd.to_datetime(start_date_str)
        self.end_date = pd.to_datetime(end_date_str)
        
        # Calculate working days
        self.working_days = np.busday_count(
            self.start_date.strftime('%Y-%m-%d'),
            (self.end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        )
        
        print(f"MetricsCalculator initialized for {start_date_str} to {end_date_str} ({self.working_days} working days).")
        
        self.load_agents()

    def load_agents(self):
        """Loads agent and manager lists from the knowledge base."""
        try:
            agents_file = os.path.join(self.kb_path, 'agents.md')
            with open(agents_file, 'r') as f:
                content = f.read()
            
            # Simple regex to pull from markdown table
            # Updated to handle spaces in names
            pattern = re.compile(r"\| ([\w\s]+) \| ([\w\.@]+) \| (\w+) \| (\w+) \|")
            matches = pattern.findall(content)
            
            agent_data = []
            for match in matches:
                # Clean whitespace from regex groups
                agent_data.append([item.strip() for item in match])

            self.agents_df = pd.DataFrame(agent_data, columns=['Agent_Name', 'Email', 'Role', 'Status'])
            
            self.active_agents_df = self.agents_df[
                (self.agents_df['Status'] == 'Active') & 
                (self.agents_df['Role'] == 'Agent')
            ].copy()
            
            self.active_managers_df = self.agents_df[
                (self.agents_df['Status'] == 'Active') & 
                (self.agents_df['Role'] == 'Manager')
            ].copy()
            
            self.active_agent_emails = self.active_agents_df['Email'].tolist()
            self.active_manager_emails = self.active_managers_df['Email'].tolist()
            
            print(f"Loaded {len(self.active_agents_df)} active agents.")
            print(f"Loaded {len(self.active_manager_emails)} active managers.")
            
        except Exception as e:
            print(f"Error loading agents from KB: {e}")
            self.active_agents_df = pd.DataFrame(columns=['Agent_Name', 'Email', 'Role', 'Status'])
            self.active_agent_emails = []
            self.active_manager_emails = []

    def _clean_phone(self, series):
        """Utility to clean phone numbers for matching."""
        return series.astype(str).str.replace(r'^\+1', '', regex=True).str.replace(r'[^0-9]', '', regex=True)

    def process_call_logs(self, file_path):
        """Processes the call log CSV for all call metrics."""
        print("Processing Call Logs...")
        try:
            df = pd.read_csv(file_path, low_memory=False)
            df['date_started'] = pd.to_datetime(df['date_started'])
            
            # Filter for time period
            df_filtered = df[
                (df['date_started'] >= self.start_date) & 
                (df['date_started'] < self.end_date + pd.Timedelta(days=1))
            ].copy()
            
            # --- 1. Basic Call Aggregates ---
            df_filtered['is_inbound_conv'] = (df_filtered['direction'] == 'inbound') & (df_filtered['talk_duration'] > 0)
            df_filtered['is_outbound_conv'] = (df_filtered['direction'] == 'outbound') & (df_filtered['talk_duration'] > 0)
            df_filtered['is_dial'] = (df_filtered['direction'] == 'outbound')
            
            calls_agg = df_filtered.groupby('email').agg(
                Dials=('is_dial', 'sum'),
                Inbound_Conversations=('is_inbound_conv', 'sum'),
                Outbound_Conversations=('is_outbound_conv', 'sum'),
                Total_Conversations=('is_inbound_conv', 'sum') + ('is_outbound_conv', 'sum'),
                Talk_Time_Seconds=('talk_duration', 'sum')
            ).reset_index()
            
            calls_agg['Talk_Time_Hours'] = calls_agg['Talk_Time_Seconds'] / 3600
            calls_agg['Avg_Daily_Inbound_Calls'] = calls_agg['Inbound_Conversations'] / self.working_days
            calls_agg['Avg_Daily_Outbound_Calls'] = calls_agg['Outbound_Conversations'] / self.working_days
            
            # --- 2. 2nd Voice Call Logic (REMOVED) ---
            
            # --- 3. Josh's (Manager) 2nd Voice Stats (REMOVED) ---
            
            # Return basic call metrics
            return calls_agg.rename(columns={'email': 'Email'})

        except Exception as e:
            print(f"Error processing Call Logs: {e}")
            return pd.DataFrame()

    def process_rescissions(self, file_paths):
        """Processes multiple rescission reports for the period."""
        print("Processing Rescission Reports...")
        try:
            df_list = []
            for f in file_paths:
                temp_df = pd.read_csv(f)
                if 'Month' in temp_df.columns:
                    temp_df['Month'] = pd.to_datetime(temp_df['Month']).dt.strftime('%Y-%m')
                df_list.append(temp_df)
                
            df = pd.concat(df_list, ignore_index=True)
            df = df.drop_duplicates()
            
            df = df[df['Agent_Name'] != 'Advantage First TOTAL'].copy()
            
            # Filter for months in range
            start_month = self.start_date.strftime('%Y-%m')
            end_month = self.end_date.strftime('%Y-%m')
            months_in_range = pd.date_range(start_month, end_month, freq='MS').strftime('%Y-%m').tolist()
            
            df_filtered = df[df['Month'].isin(months_in_range)]
            
            cols_to_sum = ['Enrollments', 'Rescissions', 'First_Drafts']
            for col in cols_to_sum:
                if col in df_filtered.columns:
                    df_filtered[col] = df_filtered[col].astype(str).str.replace(r'[$,%]', '', regex=True)
                    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce').fillna(0)
                else:
                    df_filtered[col] = 0 # Add col if missing
            
            metrics_df = df_filtered.groupby('Agent_Name')[cols_to_sum].sum().reset_index()
            return metrics_df

        except Exception as e:
            print(f"Error processing Rescission Reports: {e}")
            return pd.DataFrame()

    def process_daily_enrollments(self, file_path):
        """Processes the daily enrollment report for debt."""
        print("Processing Daily Enrollment Report...")
        try:
            df = pd.read_csv(file_path)
            df = df.rename(columns={'AFF_Agents': 'Agent_Name'})
            df['Enrollment_Date'] = pd.to_datetime(df['Enrollment_Date'])
            
            df_filtered = df[
                (df['Enrollment_Date'] >= self.start_date) & 
                (df['Enrollment_Date'] < self.end_date + pd.Timedelta(days=1))
            ].copy()
            
            df_filtered['Original_Enrolled_Debt'] = df_filtered['Original_Enrolled_Debt'].astype(str).str.replace(r'[$,]', '', regex=True)
            df_filtered['Original_Enrolled_Debt'] = pd.to_numeric(df_filtered['Original_Enrolled_Debt'], errors='coerce').fillna(0)
            
            # 1. Aggregate for main report
            debt_agg = df_filtered.groupby('Agent_Name').agg(
                Total_Enrolled_Debt=('Original_Enrolled_Debt', 'sum'),
                Average_Enrolled_Debt=('Original_Enrolled_Debt', 'mean')
            ).reset_index()
            
            # 2. Match for Josh's stats (REMOVED)

            return debt_agg
            
        except Exception as e:
            print(f"Error processing Daily Enrollments: {e}")
            return pd.DataFrame()

    def process_retention_report(self, file_path):
        """Processes the retention report for cleared deals."""
        print("Processing Retention Report...")
        try:
            df = pd.read_csv(file_path, low_memory=False)
            df = df.rename(columns={'AFF_Agents': 'Agent_Name'})
            df['Enrollment_Date'] = pd.to_datetime(df['Enrollment_Date'])
            
            df_filtered = df[
                (df['Enrollment_Date'] >= self.start_date) & 
                (df['Enrollment_Date'] < self.end_date + pd.Timedelta(days=1))
            ].copy()
            
            # Filter for Cleared Deals
            if 'First_Draft_Status' not in df_filtered.columns:
                 print("Warning: 'First_Draft_Status' column not found in retention report.")
                 return pd.DataFrame(columns=['Agent_Name', 'Cleared_Deals', 'Cleared_Debt_Load'])

            cleared_deals_df = df_filtered[df_filtered['First_Draft_Status'] == 'Cleared'].copy()
            
            cleared_deals_df['Original_Enrolled_Debt'] = cleared_deals_df['Original_Enrolled_Debt'].astype(str).str.replace(r'[$,]', '', regex=True)
            cleared_deals_df['Original_Enrolled_Debt'] = pd.to_numeric(cleared_deals_df['Original_Enrolled_Debt'], errors='coerce').fillna(0)
            
            cleared_agg = cleared_deals_df.groupby('Agent_Name').agg(
                Cleared_Deals=('CRM_ID', 'nunique'),
                Cleared_Debt_Load=('Original_Enrolled_Debt', 'sum')
            ).reset_index()
            
            return cleared_agg

        except Exception as e:
            print(f"Error processing Retention Report: {e}")
            return pd.DataFrame()

    def generate_final_report(self, file_paths):
        """Orchestrates all processing and merges data into a final report."""
        
        call_metrics = self.process_call_logs(file_paths['calls'])
        rescission_metrics = self.process_rescissions(file_paths['rescissions'])
        enrollment_metrics = self.process_daily_enrollments(file_paths['daily_enrollment'])
        retention_metrics = self.process_retention_report(file_paths['retention'])
        
        # Start with the active agent list
        final_df = self.active_agents_df.copy()
        
        # Merge all data sources
        final_df = pd.merge(final_df, call_metrics, on='Email', how='left')
        final_df = pd.merge(final_df, rescission_metrics, on='Agent_Name', how='left')
        final_df = pd.merge(final_df, enrollment_metrics, on='Agent_Name', how='left')
        final_df = pd.merge(final_df, retention_metrics, on='Agent_Name', how='left')
        
        # Fill NaNs
        fill_zero_cols = [
            'Dials', 'Inbound_Conversations', 'Outbound_Conversations', 'Total_Conversations',
            'Talk_Time_Hours', 'Avg_Daily_Inbound_Calls', 'Avg_Daily_Outbound_Calls',
            'Enrollments', 'Rescissions', 'First_Drafts',
            'Total_Enrolled_Debt', 'Average_Enrolled_Debt', 'Cleared_Deals', 'Cleared_Debt_Load'
        ]
        for col in fill_zero_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna(0)
            else:
                final_df[col] = 0
        
        # Calculate final rates
        final_df['Rescission_Rate'] = (final_df['Rescissions'] / final_df['Enrollments'] * 100)
        final_df['First_Draft_Rate'] = (final_df['First_Drafts'] / final_df['Enrollments'] * 100)
        final_df['Conv_per_Dial'] = (final_df['Outbound_Conversations'] / final_df['Dials'] * 100)
        final_df['Enr_per_Conv'] = (final_df['Enrollments'] / final_df['Total_Conversations'] * 100)
        
        final_df = final_df.replace([np.inf, -np.inf], 0).fillna(0)
        
        return final_df

# Main execution block
if __name__ == "__main__":
    # This block allows the script to be run from the command line.
    # The AI will import this class and call its methods.
    
    # --- CONFIGURATION ---
    # In a real run, these values would be provided by the AI.
    KB_PATH = '../knowledge_base'
    START_DATE = '2025-08-01'
    END_DATE = '2025-10-29'
    
    # File paths (using examples here)
    FILE_PATHS = {
        "calls": "../data_examples/call_logs_example.csv",
        "rescissions": ["../data_examples/rescission_report_example.csv"],
        "daily_enrollment": "../data_examples/daily_enrollment_example.csv",
        "retention": "../data_examples/retention_report_example.csv"
    }
    
    # --- EXECUTION ---
    print("--- RUNNING METRICS CALCULATOR (TEST MODE) ---")
    
    calculator = MetricsCalculator(KB_PATH, START_DATE, END_DATE)
    
    if not calculator.active_agents_df.empty:
        report = calculator.generate_final_report(FILE_PATHS)
        
        print("\n--- FINAL REPORT ---")
        print(report.to_markdown(index=False, floatfmt=".2f"))
        
        # --- JOSH G. 2ND VOICE STATS (REMOVED) ---
        
    else:
        print("Could not run report: No active agents found in knowledge base.")