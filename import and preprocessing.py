import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ExcelDataProcessor:
    def __init__(self, data_directory='.'):
        """
        Initialize the Excel Data Processor
        
        Args:
            data_directory (str): Directory containing Excel files
        """
        self.data_directory = data_directory
        # File names without extensions - will be detected automatically
        self.file_mapping = {}
        self.years = [2015, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        self.raw_data = {}
        self.combined_data = None
        self.processed_data = None
    
    def scan_directory(self):
        """
        Scan the data directory to find available Excel files and update file mapping
        """
        print(f"Scanning directory: {os.path.abspath(self.data_directory)}")
        
        if not os.path.exists(self.data_directory):
            print(f"Directory {self.data_directory} does not exist!")
            return {}
        
        available_files = []
        for filename in os.listdir(self.data_directory):
            if filename.lower().endswith(('.xlsx', '.xls')):
                available_files.append(filename)
        
        print(f"Found Excel files: {available_files}")
        
        # Try to match files to years
        updated_mapping = {}
        for year in self.years:
            # Look for files that start with the year
            for filename in available_files:
                # Extract the year from filename
                file_year_str = filename.split('.')[0].strip()  # Remove extension and spaces
                try:
                    file_year = int(file_year_str)
                    if file_year == year:
                        updated_mapping[year] = filename
                        break
                except ValueError:
                    continue
        
        if updated_mapping:
            self.file_mapping = updated_mapping
            self.years = sorted(list(self.file_mapping.keys()))
            print(f"Updated file mapping: {self.file_mapping}")
        else:
            print("No matching files found for expected years!")
        
        return self.file_mapping
    
    def load_excel_files(self, sheet_name=0):
        """
        Load Excel files for all specified years
        
        Args:
            sheet_name: Sheet name or index to read (default: 0 for first sheet)
        """
        print("Loading Excel files...")
        
        # First scan directory to update file mapping
        self.scan_directory()
        
        for year, filename in self.file_mapping.items():
            file_path = os.path.join(self.data_directory, filename)
            
            try:
                if os.path.exists(file_path):
                    # Read Excel file
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Add year column to identify data source
                    df['year'] = year
                    
                    self.raw_data[year] = df
                    print(f"✓ Loaded {filename} - Shape: {df.shape}")
                    
                else:
                    print(f"⚠ File not found: {file_path}")
                    
            except Exception as e:
                print(f"✗ Error loading {filename}: {str(e)}")
        
        print(f"\nSuccessfully loaded {len(self.raw_data)} files")
        return self.raw_data
    
    def inspect_first_file(self):
        """
        Inspect the first loaded file to understand the data structure
        """
        if not self.raw_data:
            print("No data loaded yet. Please run load_excel_files() first.")
            return
        
        first_year = min(self.raw_data.keys())
        first_df = self.raw_data[first_year]
        
        print(f"\n=== INSPECTION OF {first_year} DATA ===")
        print(f"Shape: {first_df.shape}")
        print(f"\nColumn names:")
        for i, col in enumerate(first_df.columns):
            print(f"  {i}: '{col}' (dtype: {first_df[col].dtype})")
        
        print(f"\nFirst 5 rows:")
        print(first_df.head())
        
        print(f"\nData types:")
        print(first_df.dtypes)
        
        print(f"\nMissing values:")
        missing = first_df.isnull().sum()
        print(missing[missing > 0])
        
        return first_df
    
    def combine_data(self):
        """
        Combine all loaded Excel files into a single DataFrame
        """
        if not self.raw_data:
            raise ValueError("No data loaded. Please run load_excel_files() first.")
        
        print("Combining data from all years...")
        
        # Check if all dataframes have similar structure
        column_sets = []
        for year, df in self.raw_data.items():
            column_sets.append(set(df.columns))
        
        # Find common columns
        common_columns = set.intersection(*column_sets) if column_sets else set()
        all_columns = set.union(*column_sets) if column_sets else set()
        
        print(f"Common columns across all files: {len(common_columns)}")
        print(f"Total unique columns: {len(all_columns)}")
        
        if len(common_columns) < len(all_columns):
            print("⚠ Warning: Not all files have the same columns!")
            different_cols = all_columns - common_columns
            print(f"Columns that differ: {different_cols}")
        
        # Combine all DataFrames
        dataframes = list(self.raw_data.values())
        self.combined_data = pd.concat(dataframes, ignore_index=True, sort=False)
        
        print(f"✓ Combined data shape: {self.combined_data.shape}")
        print(f"✓ Years included: {sorted(self.combined_data['year'].unique())}")
        
        return self.combined_data
    
    def preprocess_data(self):
        """
        Perform comprehensive data preprocessing
        """
        if self.combined_data is None:
            raise ValueError("No combined data available. Please run combine_data() first.")
        
        print("Starting data preprocessing...")
        
        # Create a copy for processing
        self.processed_data = self.combined_data.copy()
        
        # 1. Basic info about the dataset
        print(f"\nOriginal data shape: {self.processed_data.shape}")
        print(f"Columns: {len(self.processed_data.columns)} columns")
        
        # Show first few column names to check for issues
        print(f"First 10 columns: {list(self.processed_data.columns[:10])}")
        
        # 2. Clean column names (remove extra spaces, standardize)
        print("\n--- Cleaning Column Names ---")
        original_columns = self.processed_data.columns.tolist()
        
        # Clean column names
        cleaned_columns = []
        for col in original_columns:
            if isinstance(col, str):
                # Remove leading/trailing spaces and replace multiple spaces with single space
                cleaned_col = ' '.join(str(col).strip().split())
                cleaned_columns.append(cleaned_col)
            else:
                cleaned_columns.append(col)
        
        # Check for duplicate column names after cleaning
        if len(cleaned_columns) != len(set(cleaned_columns)):
            print("⚠ Warning: Duplicate column names found after cleaning!")
            duplicate_cols = [col for col in set(cleaned_columns) if cleaned_columns.count(col) > 1]
            print(f"Duplicate columns: {duplicate_cols}")
            
            # Handle duplicates by adding suffixes
            seen = {}
            final_columns = []
            for col in cleaned_columns:
                if col in seen:
                    seen[col] += 1
                    final_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    final_columns.append(col)
            
            self.processed_data.columns = final_columns
            print("✓ Fixed duplicate column names by adding suffixes")
        else:
            self.processed_data.columns = cleaned_columns
        
        # 3. Handle missing values
        print("\n--- Handling Missing Values ---")
        missing_counts = self.processed_data.isnull().sum()
        missing_percent = (missing_counts / len(self.processed_data) * 100).round(2)
        
        missing_info = pd.DataFrame({
            'Missing_Count': missing_counts,
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Count', ascending=False)
        
        columns_with_missing = missing_info[missing_info['Missing_Count'] > 0]
        if not columns_with_missing.empty:
            print("Columns with missing values:")
            print(columns_with_missing.head(10))  # Show top 10
        else:
            print("No missing values found!")
        
        # Fill numeric columns with median
        numeric_columns = self.processed_data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'year']
        
        for col in numeric_columns:
            missing_count = self.processed_data[col].isnull().sum()
            if missing_count > 0:
                median_val = self.processed_data[col].median()
                self.processed_data[col].fillna(median_val, inplace=True)
                print(f"  Filled {missing_count} missing values in '{col}' with median: {median_val}")
        
        # Fill categorical columns with mode
        categorical_columns = self.processed_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            missing_count = self.processed_data[col].isnull().sum()
            if missing_count > 0:
                mode_val = self.processed_data[col].mode()
                if len(mode_val) > 0:
                    self.processed_data[col].fillna(mode_val[0], inplace=True)
                    print(f"  Filled {missing_count} missing values in '{col}' with mode: '{mode_val[0]}'")
                else:
                    # If no mode available, fill with 'Unknown' or similar
                    self.processed_data[col].fillna('غير محدد', inplace=True)
                    print(f"  Filled {missing_count} missing values in '{col}' with 'غير محدد' (no mode available)")
        
        # 4. Remove duplicates
        print(f"\n--- Removing Duplicates ---")
        initial_rows = len(self.processed_data)
        self.processed_data.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(self.processed_data)
        print(f"Removed {duplicates_removed} duplicate rows")
        
        # 5. Data type optimization
        print(f"\n--- Optimizing Data Types ---")
        self.optimize_data_types()
        
        # 6. Handle outliers (using IQR method for numeric columns)
        print(f"\n--- Handling Outliers ---")
        if len(numeric_columns) > 0:
            self.handle_outliers()
        else:
            print("No numeric columns found for outlier handling")
        
        # 7. Final summary
        print(f"\n--- Preprocessing Complete ---")
        print(f"Final data shape: {self.processed_data.shape}")
        print(f"Memory usage: {self.processed_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return self.processed_data
    
    def optimize_data_types(self):
        """
        Optimize data types to reduce memory usage
        """
        memory_before = self.processed_data.memory_usage(deep=True).sum() / 1024**2
        
        for col in self.processed_data.columns:
            if col == 'year':
                continue
                
            col_type = self.processed_data[col].dtype
            
            if col_type != 'object':
                try:
                    # Optimize numeric columns
                    if col_type == 'int64':
                        if self.processed_data[col].min() >= 0:
                            if self.processed_data[col].max() < 255:
                                self.processed_data[col] = self.processed_data[col].astype('uint8')
                            elif self.processed_data[col].max() < 65535:
                                self.processed_data[col] = self.processed_data[col].astype('uint16')
                            elif self.processed_data[col].max() < 4294967295:
                                self.processed_data[col] = self.processed_data[col].astype('uint32')
                        else:
                            if (self.processed_data[col].min() > -128 and 
                                self.processed_data[col].max() < 127):
                                self.processed_data[col] = self.processed_data[col].astype('int8')
                            elif (self.processed_data[col].min() > -32768 and 
                                  self.processed_data[col].max() < 32767):
                                self.processed_data[col] = self.processed_data[col].astype('int16')
                            elif (self.processed_data[col].min() > -2147483648 and 
                                  self.processed_data[col].max() < 2147483647):
                                self.processed_data[col] = self.processed_data[col].astype('int32')
                    
                    elif col_type == 'float64':
                        self.processed_data[col] = pd.to_numeric(self.processed_data[col], downcast='float')
                except:
                    # If optimization fails, keep original type
                    pass
        
        memory_after = self.processed_data.memory_usage(deep=True).sum() / 1024**2
        print(f"Memory optimized: {memory_before:.2f} MB → {memory_after:.2f} MB")
    
    def handle_outliers(self, method='iqr'):
        """
        Handle outliers in numeric columns
        
        Args:
            method (str): Method to handle outliers ('iqr' or 'zscore')
        """
        numeric_columns = self.processed_data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'year']
        
        outliers_info = {}
        
        for col in numeric_columns:
            try:
                if method == 'iqr':
                    Q1 = self.processed_data[col].quantile(0.25)
                    Q3 = self.processed_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR > 0:  # Only proceed if IQR is meaningful
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers_mask = ((self.processed_data[col] < lower_bound) | 
                                       (self.processed_data[col] > upper_bound))
                        outliers_count = outliers_mask.sum()
                        
                        if outliers_count > 0:
                            # Cap outliers instead of removing them
                            self.processed_data[col] = np.where(self.processed_data[col] < lower_bound, 
                                                              lower_bound, self.processed_data[col])
                            self.processed_data[col] = np.where(self.processed_data[col] > upper_bound, 
                                                              upper_bound, self.processed_data[col])
                            
                            outliers_info[col] = outliers_count
            except:
                # Skip columns that cause errors
                continue
        
        total_outliers = sum(outliers_info.values())
        if total_outliers > 0:
            print(f"Capped {total_outliers} outliers across {len(outliers_info)} columns")
            if len(outliers_info) <= 10:  # Show details if not too many columns
                for col, count in outliers_info.items():
                    print(f"  {col}: {count} outliers")
        else:
            print("No outliers found or processed")
    
    def get_hr_insights(self):
        """
        Generate HR-specific insights from the processed data
        """
        if self.processed_data is None:
            print("No processed data available. Please run preprocess_data() first.")
            return None
        
        print("\n=== HR INSIGHTS ===")
        insights = {}
        
        # Gender distribution
        gender_col = None
        for col in self.processed_data.columns:
            if 'جنس' in col or 'gender' in col.lower():
                gender_col = col
                break
        
        if gender_col:
            print(f"\nGender Distribution:")
            gender_dist = self.processed_data[gender_col].value_counts()
            insights['gender_distribution'] = dict(gender_dist)
            for gender, count in gender_dist.items():
                percentage = (count / len(self.processed_data) * 100)
                print(f"  {gender}: {count} ({percentage:.1f}%)")
        
        # Turnover reasons
        reason_col = None
        for col in self.processed_data.columns:
            if 'سبب' in col and 'ترك' in col:
                reason_col = col
                break
        
        if reason_col:
            print(f"\nTop Turnover Reasons:")
            reasons = self.processed_data[reason_col].value_counts().head(10)
            insights['top_reasons'] = dict(reasons.head(5))
            for reason, count in reasons.items():
                if pd.notna(reason):
                    percentage = (count / len(self.processed_data) * 100)
                    print(f"  {reason}: {count} ({percentage:.1f}%)")
        
        # Yearly turnover trends
        print(f"\nYearly Turnover:")
        yearly_counts = self.processed_data['year'].value_counts().sort_index()
        insights['yearly_turnover'] = dict(yearly_counts)
        for year, count in yearly_counts.items():
            print(f"  {year}: {count} employees left")
        
        # Age analysis
        age_col = None
        for col in self.processed_data.columns:
            if 'عمر' in col and col != 'العمر عند الترك':
                age_col = col
                break
        
        if age_col:
            print(f"\nAge Statistics:")
            age_stats = self.processed_data[age_col].describe()
            insights['age_stats'] = {
                'mean': age_stats['mean'],
                'min': age_stats['min'],
                'max': age_stats['max'],
                'std': age_stats['std']
            }
            print(f"  Average Age: {age_stats['mean']:.1f} years")
            print(f"  Age Range: {age_stats['min']:.1f} - {age_stats['max']:.1f} years")
        
        # Department analysis
        dept_col = None
        for col in self.processed_data.columns:
            if 'إدارة' in col or 'قسم' in col or 'وحدة' in col:
                dept_col = col
                break
        
        if dept_col:
            print(f"\nDepartment Turnover:")
            dept_counts = self.processed_data[dept_col].value_counts().head(10)
            insights['department_turnover'] = dict(dept_counts.head(5))
            for dept, count in dept_counts.items():
                if pd.notna(dept):
                    percentage = (count / len(self.processed_data) * 100)
                    print(f"  {dept}: {count} ({percentage:.1f}%)")
        
        # Service length analysis
        service_col = None
        for col in self.processed_data.columns:
            if ('عدد' in col and 'سنوات' in col) or ('مدة' in col and 'عمل' in col):
                if self.processed_data[col].dtype in ['float64', 'int64']:
                    service_col = col
                    break
        
        if service_col:
            print(f"\nService Length Statistics:")
            service_stats = self.processed_data[service_col].describe()
            insights['service_stats'] = {
                'mean': service_stats['mean'],
                'median': service_stats['50%'],
                'min': service_stats['min'],
                'max': service_stats['max']
            }
            print(f"  Average Service: {service_stats['mean']:.1f} years")
            print(f"  Median Service: {service_stats['50%']:.1f} years")
            print(f"  Service Range: {service_stats['min']:.1f} - {service_stats['max']:.1f} years")
        
        insights['total_records'] = len(self.processed_data)
        insights['years_covered'] = sorted(self.processed_data['year'].unique())
        
        return insights
    
    def get_data_summary(self):
        """
        Generate comprehensive data summary
        """
        if self.processed_data is None:
            print("No processed data available. Please run preprocess_data() first.")
            return None
        
        print("\n=== DATA SUMMARY ===")
        print(f"Shape: {self.processed_data.shape}")
        print(f"Memory usage: {self.processed_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"\nYears covered: {sorted(self.processed_data['year'].unique())}")
        
        year_counts = self.processed_data['year'].value_counts().sort_index()
        print(f"Records per year:")
        for year, count in year_counts.items():
            print(f"  {year}: {count:,} records")
        
        print(f"\nData types summary:")
        dtype_counts = self.processed_data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Numeric columns summary
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric columns ({len(numeric_cols)} total):")
            print(self.processed_data[numeric_cols].describe())
        
        return {
            'shape': self.processed_data.shape,
            'columns': list(self.processed_data.columns),
            'dtypes': dict(self.processed_data.dtypes),
            'years': sorted(self.processed_data['year'].unique()),
            'records_per_year': dict(year_counts)
        }
    
    def save_processed_data(self, output_path='processed_data.xlsx'):
        """
        Save processed data to Excel file
        
        Args:
            output_path (str): Path to save the processed data
        """
        if self.processed_data is None:
            print("No processed data to save. Please run preprocess_data() first.")
            return False
        
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Use xlsxwriter engine to handle Arabic text better
            with pd.ExcelWriter(output_path, engine='xlsxwriter', options={'strings_to_urls': False}) as writer:
                self.processed_data.to_excel(writer, index=False, sheet_name='HR_Data')
                
                # Get workbook and worksheet objects
                workbook = writer.book
                worksheet = writer.sheets['HR_Data']
                
                # Add a format for Arabic text
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Format headers
                for col_num, value in enumerate(self.processed_data.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    
                # Auto-adjust column widths
                for i, col in enumerate(self.processed_data.columns):
                    # Calculate max width needed
                    max_length = max(
                        self.processed_data[col].astype(str).map(len).max(),  # Max length in column
                        len(str(col))  # Length of column name
                    )
                    # Set column width (with some padding)
                    worksheet.set_column(i, i, min(max_length + 2, 50))
            
            print(f"✓ Processed data saved to {output_path}")
            return True
        except Exception as e:
            print(f"✗ Error saving data: {str(e)}")
            # Fallback to basic Excel writer
            try:
                self.processed_data.to_excel(output_path, index=False, engine='openpyxl')
                print(f"✓ Processed data saved to {output_path} (using fallback method)")
                return True
            except Exception as e2:
                print(f"✗ Fallback also failed: {str(e2)}")
                return False


# Example usage
def main():
    """
    Main function demonstrating how to use the ExcelDataProcessor
    """
    print("=== Excel Data Processor Started ===")
    
    # Initialize processor - use current directory since files are in the same folder
    processor = ExcelDataProcessor(data_directory='.')
    
    try:
        # Load Excel files
        raw_data = processor.load_excel_files()
        
        if raw_data:
            print(f"\n✓ Successfully loaded {len(raw_data)} files")
            
            # Inspect first file to understand structure
            processor.inspect_first_file()
            
            # Combine data
            combined_data = processor.combine_data()
            
            # Preprocess data
            processed_data = processor.preprocess_data()
            
            # Get summary
            summary = processor.get_data_summary()
            
            # Get HR insights (FIXED - now this method exists!)
            insights = processor.get_hr_insights()
            
            # Save processed data
            success = processor.save_processed_data('processed_combined_data.xlsx')
            
            if success:
                print(f"\n🎉 Processing completed successfully!")
                print(f"Final dataset: {processed_data.shape[0]:,} rows × {processed_data.shape[1]} columns")
            
            return processor
            
        else:
            print("\n❌ No data files found or loaded.")
            print("Please ensure Excel files are in the same directory as this script")
            print("Looking for files: 2015.xlsx, 2017.xlsx, 2018.xlsx, 2019.xlsx, 2020.xlsx, 2021.xlsx, 2022.xlsx, 2023.xlsx, 2024.xlsx")
            return None
            
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the main function
    processor = main()