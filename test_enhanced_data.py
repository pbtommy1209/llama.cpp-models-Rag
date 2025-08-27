#!/usr/bin/env python3
"""
Test Enhanced Dataset Creation
Simple script to test the enhanced dataset creation without running the full pipeline
"""

import json
from pathlib import Path

def test_enhanced_data():
    """Test the enhanced dataset creation"""
    print("🧪 Testing Enhanced Dataset Creation...")
    
    # Test 1: Run the extractor
    try:
        import subprocess
        import sys
        
        print("📊 Creating enhanced dataset...")
        result = subprocess.run([sys.executable, "extract_sheet_data.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ Enhanced dataset created successfully!")
        
        # Test 2: Check generated files
        expected_files = [
            "data/comprehensive_faq_docs.jsonl",
            "data/comprehensive_eval_questions.jsonl",
            "data/performance_test_cases.jsonl",
            "data/comprehensive_dataset_metadata.json"
        ]
        
        print("\n📁 Checking generated files...")
        for file_path in expected_files:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.endswith('.jsonl'):
                        lines = f.readlines()
                        print(f"  ✅ {file_path}: {len(lines)} lines")
                    else:
                        data = json.load(f)
                        print(f"  ✅ {file_path}: {len(data)} entries")
            else:
                print(f"  ❌ {file_path}: Missing")
        
        # Test 3: Show sample data
        print("\n📋 Sample data from comprehensive FAQ:")
        if Path("data/comprehensive_faq_docs.jsonl").exists():
            with open("data/comprehensive_faq_docs.jsonl", 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    sample = json.loads(lines[0])
                    print(f"  Question: {sample['question'][:100]}...")
                    print(f"  Answer: {sample['answer'][:100]}...")
                    print(f"  Model: {sample['model_name']}")
                    print(f"  Category: {sample['category']}")
        
        print("\n🎉 Enhanced dataset test completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Run: python run_complete_enhanced_pipeline.py")
        print("2. Or run individual steps:")
        print("   - python scripts/preprocess.py")
        print("   - python scripts/rag_pipeline.py")
        print("   - python eval_easy.py")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Dataset creation failed: {e}")
        print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_enhanced_data()
