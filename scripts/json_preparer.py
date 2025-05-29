#!/usr/bin/env python3
"""
Excel to JSON converter for scientific papers
Converts Excel file with columns: "Study Name", "Category", "Link", "Paper Summary"
to JSON format compatible with load_documents.py
"""
import pandas as pd
import json
import os
import sys
from typing import List, Dict

def excel_to_json(excel_path: str, output_path: str = None) -> bool:
    """
    Convert Excel file to JSON format for document loading
    Supports both scientific papers and FAQ formats
    
    Args:
        excel_path: Path to the Excel file
        output_path: Path for the output JSON file (optional)
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Read Excel file
        print(f"Reading Excel file: {excel_path}")
        df = pd.read_excel(excel_path)
        
        # Auto-detect file type based on columns
        file_type = detect_file_type(df.columns)
        if file_type is None:
            return False
        
        print(f"Detected file type: {file_type}")
        
        # Convert based on file type
        if file_type == "papers":
            documents = convert_papers_to_json(df)
        elif file_type == "faq":
            documents = convert_faq_to_json(df)
        else:
            print(f"Unknown file type: {file_type}")
            return False
        
        if not documents:
            print("No valid documents found to convert")
            return False
        
        print(f"Converted {len(documents)} documents to JSON format")
        
        
        if output_path is None: 
            base_name = os.path.splitext(excel_path)[0]
            output_path = f"{base_name}.json"
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        
        print(f"JSON file saved: {output_path}")
        print(f"Ready to use with: python scripts/load_documents.py {output_path}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Excel file not found: {excel_path}")
        return False
    except Exception as e:
        print(f"Error converting Excel to JSON: {e}")
        return False

def detect_file_type(columns: List[str]) -> str:
    """
    Detect the type of Excel file based on column names
    
    Args:
        columns: List of column names from the Excel file
    
    Returns:
        str: 'papers', 'faq', or None if unknown
    """
    columns_lower = [col.lower() for col in columns]
    
    # Check for papers format
    papers_columns = ["study name", "category", "link", "paper summary"]
    if all(col in columns_lower for col in papers_columns):
        return "papers"
    
    # Check for FAQ format
    faq_columns = ["question", "answer"]
    if all(col in columns_lower for col in faq_columns):
        return "faq"
    
    print(f"Error: Unknown file format. Available columns: {list(columns)}")
    print("Supported formats:")
    print("  Papers: Study Name, Category, Link, Paper Summary")
    print("  FAQ: Question, Answer")
    return None

def convert_papers_to_json(df: pd.DataFrame) -> List[Dict]:
    """Convert papers DataFrame to JSON format"""
    documents = []
    for index, row in df.iterrows():
        # Skip rows with missing essential data
        if pd.isna(row["Study Name"]) or pd.isna(row["Paper Summary"]):
            print(f"Skipping row {index + 1}: Missing study name or summary")
            continue
        
        # Create document structure
        doc = {
            "id": f"paper_{index + 1:03d}",  # e.g., paper_001, paper_002
            "title": str(row["Study Name"]).strip(),
            "link": str(row["Link"]).strip() if not pd.isna(row["Link"]) else "",
            "content": str(row["Paper Summary"]).strip()
        }
        documents.append(doc)
    
    return documents

def convert_faq_to_json(df: pd.DataFrame) -> List[Dict]:
    """Convert FAQ DataFrame to JSON format"""
    documents = []
    for index, row in df.iterrows():
        # Skip rows with missing essential data
        if pd.isna(row["Question"]) or pd.isna(row["Answer"]):
            print(f"Skipping row {index + 1}: Missing question or answer")
            continue
        
        # Create document structure
        doc = {
            "id": f"faq_{index + 1:03d}",  # e.g., faq_001, faq_002
            "title": str(row["Question"]).strip(),
            "link": "",  # FAQ entries don't have links
            "content": str(row["Answer"]).strip()
        }
        documents.append(doc)
    
    return documents

def print_usage():
    """Print usage instructions"""
    print("Usage:")
    print("  python scripts/json_preparer.py [excel_file] [output_file]")
    print("")
    print("Examples:")
    print("  python scripts/json_preparer.py scientific_papers.xlsx")
    print("  python scripts/json_preparer.py faq.xlsx")
    print("  python scripts/json_preparer.py scientific_papers.xlsx papers.json")
    print("")
    print("Supported Excel formats:")
    print("  Papers: Study Name, Category, Link, Paper Summary")
    print("  FAQ: Question, Answer")
    print("")
    print("Output JSON structure:")
    print("  Papers: {id, title, link, content}")
    print("  FAQ: {id, title, link='', content}")

def main():
    """Main function"""
    print("=== Excel to JSON Converter ===")
    
    # Check command line arguments
    if len(sys.argv) < 2:
        # Default to scientific_papers.xlsx in same directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        excel_path = os.path.join(script_dir, "scientific_papers.xlsx")
        
        if not os.path.exists(excel_path):
            print("No Excel file specified and scientific_papers.xlsx not found in scripts directory")
            print_usage()
            return
    else:
        if sys.argv[1] in ["-h", "--help"]:
            print_usage()
            return
        excel_path = sys.argv[1]
    
    # Get output path if specified
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Convert Excel to JSON
    success = excel_to_json(excel_path, output_path)
    
    if success:
        print("=== Conversion Complete ===")
    else:
        print("=== Conversion Failed ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
