"""
Fresh Complete Dataset Loader
Downloads UCI + loads manual Mendeley files + generates A2P
"""

import pandas as pd
import requests
from pathlib import Path
import zipfile
import json
from sklearn.model_selection import train_test_split
from datetime import datetime

# Directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_uci_sms_spam():
    """Download UCI SMS Spam Collection"""
    print("\n" + "="*60)
    print("STEP 1: UCI SMS SPAM COLLECTION")
    print("="*60)
    
    sources = [
        {
            'name': 'GitHub Mirror 1',
            'url': 'https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/SMSSpamCollection.csv',
            'type': 'csv'
        },
        {
            'name': 'GitHub Mirror 2',
            'url': 'https://raw.githubusercontent.com/justmarkham/pydata-dc-2016-tutorial/master/data/sms.tsv',
            'type': 'tsv'
        }
    ]
    
    for source in sources:
        try:
            print(f"\n📥 Trying {source['name']}...")
            response = requests.get(source['url'], timeout=30)
            response.raise_for_status()
            
            from io import StringIO
            if source['type'] == 'csv':
                df = pd.read_csv(StringIO(response.text))
                if 'v1' in df.columns:
                    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
            else:  # tsv
                df = pd.read_csv(StringIO(response.text), sep='\t')
                if df.columns[0] != 'label':
                    df.columns = ['label', 'message']
            
            # Clean
            df['label'] = df['label'].str.lower().map({'ham': 'ham', 'spam': 'spam'})
            df = df[df['label'].isin(['ham', 'spam'])].copy()
            df['source'] = 'uci'
            df['dataset'] = 'uci_sms_spam'
            df['quality'] = 'high'
            df['is_smishing'] = False
            
            print(f"✅ SUCCESS! {len(df)} messages")
            print(f"   - Spam: {len(df[df['label']=='spam'])} ({len(df[df['label']=='spam'])/len(df)*100:.1f}%)")
            print(f"   - Ham: {len(df[df['label']=='ham'])} ({len(df[df['label']=='ham'])/len(df)*100:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    print("\n⚠️  All UCI sources failed, continuing without UCI data")
    return None


def load_mendeley_10191():
    """Load Dataset_10191.csv (should be in data/raw/)"""
    print("\n" + "="*60)
    print("STEP 2: MENDELEY DATASET 10191")
    print("="*60)
    
    csv_path = RAW_DIR / "Dataset_10191.csv"
    
    if not csv_path.exists():
        print(f"❌ NOT FOUND: {csv_path}")
        print("   Please copy Dataset_10191.csv to data/raw/")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded: {len(df)} messages")
        print(f"   Columns: {list(df.columns)}")
        
        # Standardize columns
        df = df.rename(columns={'LABEL': 'label', 'TEXT': 'message'})
        
        # Standardize labels
        df['label'] = df['label'].str.lower()
        
        print(f"\n   Original labels:")
        print(df['label'].value_counts())
        
        # Track smishing
        df['is_smishing'] = df['label'] == 'smishing'
        
        # Binary classification
        df['label'] = df['label'].apply(lambda x: 'spam' if x in ['spam', 'smishing'] else 'ham')
        
        # Metadata
        df['source'] = 'mendeley_10191'
        df['dataset'] = 'mendeley_10191'
        df['quality'] = 'high'
        
        print(f"\n✅ Processed:")
        print(f"   - Total: {len(df)}")
        print(f"   - Spam: {len(df[df['label']=='spam'])} ({len(df[df['label']=='spam'])/len(df)*100:.1f}%)")
        print(f"   - Ham: {len(df[df['label']=='ham'])} ({len(df[df['label']=='ham'])/len(df)*100:.1f}%)")
        print(f"   - Smishing: {df['is_smishing'].sum()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def load_mendeley_5971():
    """Load Dataset_5971 (zip or csv in data/raw/)"""
    print("\n" + "="*60)
    print("STEP 3: MENDELEY DATASET 5971")
    print("="*60)
    
    zip_path = RAW_DIR / "Dataset_5971.zip"
    csv_path = RAW_DIR / "Dataset_5971.csv"
    
    # Extract if needed
    if not csv_path.exists() and zip_path.exists():
        print("   Extracting zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        print("   ✓ Extracted")
    
    if not csv_path.exists():
        print(f"❌ NOT FOUND: {csv_path}")
        print("   Please copy Dataset_5971.zip or Dataset_5971.csv to data/raw/")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded: {len(df)} messages")
        print(f"   Columns: {list(df.columns)}")
        
        # Show first row to understand format
        print(f"\n   Sample row:")
        print(df.iloc[0])
        
        # Find label and message columns (flexible matching)
        label_col = None
        message_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'label' in col_lower or 'class' in col_lower or 'category' in col_lower:
                label_col = col
            if 'text' in col_lower or 'message' in col_lower or 'sms' in col_lower:
                message_col = col
        
        if label_col is None or message_col is None:
            print(f"\n⚠️  Could not identify columns automatically")
            print(f"   Label column: {label_col}")
            print(f"   Message column: {message_col}")
            print(f"   Please check the CSV format")
            return None
        
        # Standardize
        df = df.rename(columns={label_col: 'label', message_col: 'message'})
        
        # Standardize labels
        df['label'] = df['label'].astype(str).str.lower()
        
        print(f"\n   Original labels:")
        print(df['label'].value_counts())
        
        # Track smishing
        df['is_smishing'] = df['label'].isin(['smishing', 'phishing'])
        
        # Binary classification
        df['label'] = df['label'].apply(
            lambda x: 'spam' if x in ['spam', 'smishing', 'phishing'] else 'ham'
        )
        
        # Metadata
        df['source'] = 'mendeley_5971'
        df['dataset'] = 'mendeley_5971'
        df['quality'] = 'high'
        
        print(f"\n✅ Processed:")
        print(f"   - Total: {len(df)}")
        print(f"   - Spam: {len(df[df['label']=='spam'])} ({len(df[df['label']=='spam'])/len(df)*100:.1f}%)")
        print(f"   - Ham: {len(df[df['label']=='ham'])} ({len(df[df['label']=='ham'])/len(df)*100:.1f}%)")
        print(f"   - Smishing: {df['is_smishing'].sum()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_a2p_messages():
    """Generate comprehensive A2P legitimate messages"""
    print("\n" + "="*60)
    print("STEP 4: A2P LEGITIMATE MESSAGES")
    print("="*60)
    
    messages = []
    
    # 2FA (400)
    print("   Generating 2FA/OTP...")
    otp_templates = [
        "Your {service} verification code is {code}",
        "{code} is your {service} security code",
        "Use {code} to verify your {service} account",
    ]
    
    services = ["Google", "Apple", "Amazon", "PayPal", "Chase", "Facebook", 
                "Instagram", "Netflix", "Uber", "Microsoft"]
    
    for i in range(40):
        for template in otp_templates:
            for service in services:
                code = f"{100000 + (i * 13) % 900000}"[:6]
                msg = template.format(service=service, code=code)
                messages.append({
                    'message': msg, 'label': 'ham', 'source': 'a2p_synthetic',
                    'dataset': 'a2p_2fa', 'quality': 'high', 'a2p_type': '2fa',
                    'is_smishing': False
                })
    
    # Transactional (600)
    print("   Generating transactional...")
    trans_templates = [
        "Your {service} order #{order} has shipped",
        "Payment of ${amount} to {person} successful",
        "{service}: Appointment confirmed for {date}",
    ]
    
    for i in range(200):
        for template in trans_templates:
            msg = template.format(
                service=["Amazon", "Target", "Uber"][i % 3],
                order=f"{10000+i}", amount=f"{10+i%90}.00",
                person=["John", "Jane"][i%2], date=["Monday", "tomorrow"][i%2]
            )
            messages.append({
                'message': msg, 'label': 'ham', 'source': 'a2p_synthetic',
                'dataset': 'a2p_transactional', 'quality': 'high', 
                'a2p_type': 'transactional', 'is_smishing': False
            })
    
    # Marketing (500)
    print("   Generating marketing...")
    marketing_templates = [
        "{brand}: {percent}% off! Shop now. Reply STOP to unsubscribe",
        "{brand} Rewards: You have {points} points!",
        "Weekend Sale at {brand}: Up to {percent}% off",
    ]
    
    brands = ["Target", "Nike", "Sephora", "Best Buy", "Macy's"]
    
    for i in range(167):
        for template in marketing_templates:
            msg = template.format(
                brand=brands[i%len(brands)], 
                percent=[20,30,40,50][i%4],
                points=100+i*10
            )
            messages.append({
                'message': msg, 'label': 'ham', 'source': 'a2p_synthetic',
                'dataset': 'a2p_marketing', 'quality': 'high',
                'a2p_type': 'marketing', 'is_smishing': False
            })
    
    # Alerts (500)
    print("   Generating alerts...")
    alert_templates = [
        "Appointment reminder: {service} tomorrow at {time}",
        "Weather Alert: {condition} expected tonight",
        "{service} bill due: ${amount} due by {date}",
    ]
    
    for i in range(167):
        for template in alert_templates:
            msg = template.format(
                service=["Dentist", "Doctor"][i%2],
                time=["2pm", "10am"][i%2],
                condition=["Rain", "Snow"][i%2],
                amount=[50, 100][i%2],
                date=["Friday", "Monday"][i%2]
            )
            messages.append({
                'message': msg, 'label': 'ham', 'source': 'a2p_synthetic',
                'dataset': 'a2p_alerts', 'quality': 'high',
                'a2p_type': 'alerts', 'is_smishing': False
            })
    
    df = pd.DataFrame(messages)
    print(f"✅ Generated {len(df)} A2P messages")
    print(f"   - 2FA: {len(df[df['a2p_type']=='2fa'])}")
    print(f"   - Transactional: {len(df[df['a2p_type']=='transactional'])}")
    print(f"   - Marketing: {len(df[df['a2p_type']=='marketing'])}")
    print(f"   - Alerts: {len(df[df['a2p_type']=='alerts'])}")
    
    return df


def combine_and_split(datasets):
    """Combine all datasets and create splits"""
    print("\n" + "="*60)
    print("COMBINING ALL DATASETS")
    print("="*60)
    
    # Combine
    combined = pd.concat(datasets, ignore_index=True)
    print(f"\nBefore dedup: {len(combined)} messages")
    
    # Remove duplicates
    combined = combined.drop_duplicates(subset=['message'], keep='first')
    print(f"After dedup:  {len(combined)} messages")
    
    # Ensure is_smishing exists
    if 'is_smishing' not in combined.columns:
        combined['is_smishing'] = False
    
    # Stats
    print(f"\n📊 FINAL DATASET:")
    print(f"   Total: {len(combined)}")
    print(f"   - Spam: {len(combined[combined['label']=='spam'])} ({len(combined[combined['label']=='spam'])/len(combined)*100:.1f}%)")
    print(f"   - Ham: {len(combined[combined['label']=='ham'])} ({len(combined[combined['label']=='ham'])/len(combined)*100:.1f}%)")
    print(f"   - Smishing: {combined['is_smishing'].sum()}")
    
    print(f"\n   By source:")
    for source in combined['source'].value_counts().items():
        print(f"   - {source[0]}: {source[1]}")
    
    # Stratified split
    train_df, temp_df = train_test_split(
        combined, test_size=0.3, stratify=combined['label'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
    )
    
    print(f"\n📈 Splits:")
    print(f"   Train: {len(train_df):5d} (Spam: {len(train_df[train_df['label']=='spam']):4d}, Ham: {len(train_df[train_df['label']=='ham']):4d})")
    print(f"   Val:   {len(val_df):5d} (Spam: {len(val_df[val_df['label']=='spam']):4d}, Ham: {len(val_df[val_df['label']=='ham']):4d})")
    print(f"   Test:  {len(test_df):5d} (Spam: {len(test_df[test_df['label']=='spam']):4d}, Ham: {len(test_df[test_df['label']=='ham']):4d})")
    
    # Save
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    combined.to_csv(PROCESSED_DIR / "combined.csv", index=False)
    
    # Summary
    summary = {
        'created_at': datetime.now().isoformat(),
        'total_messages': len(combined),
        'spam_count': int(len(combined[combined['label']=='spam'])),
        'ham_count': int(len(combined[combined['label']=='ham'])),
        'smishing_count': int(combined['is_smishing'].sum()),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
    }
    
    with open(PROCESSED_DIR / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Saved to {PROCESSED_DIR}/")
    return combined


def main():
    print("="*60)
    print("FRESH COMPLETE DATASET PREPARATION")
    print("="*60)
    
    datasets = []
    
    # 1. UCI
    uci = download_uci_sms_spam()
    if uci is not None:
        datasets.append(uci)
    
    # 2. Mendeley 10191
    m10191 = load_mendeley_10191()
    if m10191 is not None:
        datasets.append(m10191)
    
    # 3. Mendeley 5971
    m5971 = load_mendeley_5971()
    if m5971 is not None:
        datasets.append(m5971)
    
    # 4. A2P
    a2p = generate_a2p_messages()
    datasets.append(a2p)
    
    # Combine
    if len(datasets) > 0:
        combined = combine_and_split(datasets)
        
        print("\n" + "="*60)
        print("✅ COMPLETE!")
        print("="*60)
        print(f"\nTotal: {len(combined)} messages ready for training")
        print(f"\nNext: python src/training/train_all_models.py")
    else:
        print("\n❌ No datasets loaded!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
