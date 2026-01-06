# Manual Dataset Download Instructions

If the automatic downloader can't fetch the Mendeley datasets, you can download them manually and add them to the project.

## Mendeley Datasets to Download

### Dataset 1: SMS Phishing Dataset for Machine Learning
**URL**: https://data.mendeley.com/datasets/f45bkkt8pr/1

**How to download:**
1. Visit the URL
2. Click "Download" button (may need free Mendeley account)
3. You'll get a file (likely CSV or zip)
4. Save it as `mendeley_phishing_1.csv` in your project's `data/raw/` folder

**Expected format:**
- Should have columns like: `message`, `label` (or similar)
- Labels should be: spam/ham or phishing/legitimate

### Dataset 2: SMS Spam Collection
**URL**: https://data.mendeley.com/datasets/vmg875v4xs/1

**How to download:**
1. Visit the URL
2. Click "Download" button
3. Save as `mendeley_spam_2.csv` in `data/raw/`

**Expected format:**
- Columns: message, label/category/type
- Labels: spam/ham

## After Manual Download

Once you have the files in `data/raw/`, run this script to integrate them:

```python
# integrate_manual_datasets.py
import pandas as pd
from pathlib import Path

data_dir = Path("data")
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"

datasets = []

# Load existing processed data
existing = pd.read_csv(processed_dir / "combined.csv")
datasets.append(existing)

# Load Mendeley Dataset 1
try:
    df1 = pd.read_csv(raw_dir / "mendeley_phishing_1.csv")
    
    # Standardize columns (adjust based on actual columns)
    if 'Label' in df1.columns:
        df1 = df1.rename(columns={'Label': 'label', 'Text': 'message'})
    elif 'label' not in df1.columns and 'message' not in df1.columns:
        print("⚠️  Dataset 1: Check column names. Expected 'label' and 'message'")
        print(f"   Found: {df1.columns.tolist()}")
    
    # Standardize labels
    df1['label'] = df1['label'].str.lower()
    df1['label'] = df1['label'].replace({
        'phishing': 'spam',
        'smishing': 'spam',
        'legitimate': 'ham'
    })
    
    df1['source'] = 'mendeley_1_manual'
    df1['dataset'] = 'mendeley_phishing_1'
    df1['quality'] = 'high'
    
    print(f"✅ Loaded Mendeley Dataset 1: {len(df1)} messages")
    datasets.append(df1)
    
except FileNotFoundError:
    print("⚠️  Mendeley Dataset 1 not found at data/raw/mendeley_phishing_1.csv")
except Exception as e:
    print(f"❌ Error loading Mendeley Dataset 1: {e}")

# Load Mendeley Dataset 2
try:
    df2 = pd.read_csv(raw_dir / "mendeley_spam_2.csv")
    
    # Standardize columns
    if 'Category' in df2.columns:
        df2 = df2.rename(columns={'Category': 'label', 'Message': 'message'})
    
    df2['label'] = df2['label'].str.lower()
    df2['label'] = df2['label'].replace({'legitimate': 'ham'})
    
    df2['source'] = 'mendeley_2_manual'
    df2['dataset'] = 'mendeley_spam_2'
    df2['quality'] = 'high'
    
    print(f"✅ Loaded Mendeley Dataset 2: {len(df2)} messages")
    datasets.append(df2)
    
except FileNotFoundError:
    print("⚠️  Mendeley Dataset 2 not found at data/raw/mendeley_spam_2.csv")
except Exception as e:
    print(f"❌ Error loading Mendeley Dataset 2: {e}")

# Combine all datasets
if len(datasets) > 1:
    combined = pd.concat(datasets, ignore_index=True)
    combined = combined.drop_duplicates(subset=['message'], keep='first')
    
    print(f"\n📊 Combined Dataset: {len(combined)} messages")
    print(f"   - Spam: {len(combined[combined['label']=='spam'])}")
    print(f"   - Ham: {len(combined[combined['label']=='ham'])}")
    
    # Create new splits
    from sklearn.model_selection import train_test_split
    
    train_df, temp_df = train_test_split(
        combined, test_size=0.3, stratify=combined['label'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
    )
    
    # Save
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    combined.to_csv(processed_dir / "combined.csv", index=False)
    
    print(f"\n✅ Updated datasets saved!")
    print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
else:
    print("\n⚠️  No new datasets to integrate")
```

Save this as `integrate_manual_datasets.py` in your project root and run:

```bash
python integrate_manual_datasets.py
```

## Alternative: Direct CSV Upload

If the Mendeley files are in a different format, you can also:

1. Open the downloaded file in Excel/Google Sheets
2. Ensure it has these columns: `message`, `label`
3. Make sure labels are `spam` or `ham`
4. Save as CSV with these exact column names
5. Run the integration script above

## Verifying the Data

After integration, check the data:

```bash
# Check total messages
wc -l data/processed/combined.csv

# View first few lines
head -20 data/processed/train.csv

# Check label distribution
python -c "
import pandas as pd
df = pd.read_csv('data/processed/combined.csv')
print(df['label'].value_counts())
print(f'\nTotal: {len(df)} messages')
"
```

## Expected Dataset Sizes

| Dataset | Expected Size |
|---------|--------------|
| UCI SMS Spam | 5,574 messages |
| Mendeley Phishing 1 | 1,000-5,000 messages |
| Mendeley Spam 2 | 1,000-3,000 messages |
| A2P Synthetic | 2,000 messages |
| **Total** | **9,000-15,000 messages** |

## Troubleshooting

### "Column not found" error
- Check actual column names: `df.columns.tolist()`
- Update the rename() call to match

### "Label values not recognized"
- Check unique labels: `df['label'].unique()`
- Add mapping in the replace() call

### Duplicate messages
- The script automatically removes duplicates
- Keeps the first occurrence (higher quality source)

### Wrong file format
- Make sure file is CSV (comma-separated)
- Try opening in Excel and saving as CSV UTF-8

## Need Help?

If you're having trouble:
1. Share the first few lines of the CSV file
2. Share the column names
3. Share any error messages

The automatic downloader should work in most cases, but these manual instructions are here as a backup!
